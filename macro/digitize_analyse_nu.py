import ROOT as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Import LogNorm for log scale colorbars
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import seaborn as sns
import argparse
import uproot
from rootpyPickler import Unpickler
import shipDet_conf
import SciFiMapping
import logging
import argparse
import os
import glob
import yaml
import pickle

sns.set_style("whitegrid")




def cluster_scifi_hits(df: pd.DataFrame, *, with_qdc: bool = False) -> pd.DataFrame:
        """
        Extract SciFi hits (layer_type<2), cluster per event/layer,
        and compute exactly the same quantities as your C++ sndCluster.
        """
        sci = df[df['layer_type'] < 2].copy()
        clusters = []

        # group by event, layer_id, and layer_type
        group_cols = ['event_id', 'layer_id', 'layer_type']
        for keys, grp in sci.groupby(group_cols, sort=False):
            evt, layer_id, lt = keys
            # sort by the true SciFi index
            grp = grp.sort_values('phys_idx').reset_index(drop=True)
            phys = grp['phys_idx'].to_numpy()

            # find breaks where adjacency is lost
            breaks = np.nonzero(np.diff(phys) != 1)[0]
            starts = np.concatenate(([0], breaks + 1))
            ends   = np.concatenate((breaks, [len(phys) - 1]))

            for s, e in zip(starts, ends):
                block = grp.iloc[s:e+1]
                first_idx = int(block['phys_idx'].iat[0])
                N         = len(block)

                # weights
                if with_qdc:
                    w = block['Signal']
                else:
                    w = pd.Series(1.0, index=block.index)

                total_w = w.sum()
                mean_x  = (block['x_digi'] * w).sum() / total_w
                mean_y  = (block['y_digi'] * w).sum() / total_w
                t_min   = block['Time'].min()

                clusters.append({
                    'event_id':   evt,
                    'layer_id':   layer_id,
                    'layer_type': lt,
                    'first_idx':  first_idx,
                    'N':          N,
                    'x_digi':     mean_x,
                    'y_digi':     mean_y,
                    'Signal':     total_w,
                    'Time':       t_min
                })

        return pd.DataFrame(clusters)


from sklearn.cluster import DBSCAN

def cluster_dbscan(df: pd.DataFrame,
                   *,
                   eps: float = 20.,
                   min_samples: int = 1,
                   with_qdc: bool = True,
                   use_time: bool = False) -> pd.DataFrame:
    """
    Runs DBSCAN on each (event_id, layer_id, layer_type) SciFi group,
    clustering either in 1D (phys_idx) or 2D (phys_idx, time) space,
    then summarizes each non-noise cluster like sndCluster.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['event_id','layer_id','layer_type','phys_idx',
        'x_digi','y_digi','Signal','Time'].
    eps : float
        The maximum distance for two points to be considered neighbors.
        In 1D this is in channel‐units; in 2D this applies to both dimensions.
    min_samples : int
        Minimum number of hits to form a cluster (others are labeled noise).
    with_qdc : bool
        If True, weight positions/energy by Signal; otherwise use unit weights.
    use_time : bool
        If True, include time as a second clustering dimension (scaled to phys_idx).

    Returns
    -------
    pd.DataFrame
        One row per cluster, columns:
        ['event_id','layer_id','layer_type','cluster_lbl',
         'first_idx','N','mean_x','mean_y','energy','time']
        where 'time' = min(6.25 * Time).
    """
    # 1) Keep only SciFi hits
    sci = df[df['layer_type'] < 2].copy()

    # 2) If using time, compute a global scale so time and phys_idx ranges match
    if use_time:
        max_phys = sci['phys_idx'].max()
        max_time = sci['Time'].max()
        time_scale = (max_phys / max_time) if max_time > 0 else 1.0

    # 3) Prepare a column for DBSCAN labels
    sci['cluster_lbl'] = -1

    # 4) Run DBSCAN per (event_id, layer_id, layer_type)
    for (evt, layer_id, lt), sub in sci.groupby(
            ['event_id','layer_id','layer_type'], sort=False):
        if use_time:
            # build 2D feature array: [phys_idx, time_scaled]
            t_scaled = sub['Time'].to_numpy() * time_scale
            X = np.column_stack((sub['phys_idx'].to_numpy(), t_scaled))
        else:
            # 1D feature array: [phys_idx]
            X = sub[['phys_idx']].to_numpy()

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        sci.loc[sub.index, 'cluster_lbl'] = labels

    # 5) Summarize each non-noise cluster
    clusters = []
    mask = sci['cluster_lbl'] >= 0
    for (evt, layer_id, lt, lbl), grp in sci[mask].groupby(
            ['event_id','layer_id','layer_type','cluster_lbl'],
            sort=False):
        first_idx = int(grp['phys_idx'].min())
        N         = len(grp)
        # choose weights
        if with_qdc:
            w = grp['Signal']
        else:
            w = pd.Series(1.0, index=grp.index)

        total_w = w.sum()
        mean_x  = (grp['x_digi'] * w).sum() / total_w
        mean_y  = (grp['y_digi'] * w).sum() / total_w
        t_min     = grp['Time'].min()

        clusters.append({
            'event_id':   evt,
            'layer_id':   layer_id,
            'layer_type': lt,
            'first_idx':  first_idx,
            'N':          N,
            'x_digi':     mean_x,
            'y_digi':     mean_y,
            'Signal':     total_w,
            'Time':       t_min
        })

    return pd.DataFrame(clusters)


class FairShipAnalyzer:
    def __init__(self, file_list, detector_properties, fiber_dimensions):
        if isinstance(file_list, str):
            file_list = [file_list]
        self.file_list = file_list
        self.chain = self._create_chain()
        self.detector_properties = detector_properties
        # self.set_fiber_dimensions(detector_properties, fiber_dimensions)

    def read_geo(self, geofile):
        """
        Load geometry and build the SciFi mapping, keeping all ROOT/FairRoot C++ objects alive
        so they can still be used later in read_digi().
        """

        # --- open & keep the geometry file -------------------------------------------------
        self.fgeo = r.TFile.Open(geofile)
        if not self.fgeo or self.fgeo.IsZombie():
            raise IOError(f"Cannot open geofile {geofile}")

        upkl = Unpickler(self.fgeo)
        self.ship_geo = upkl.load("ShipGeo")

        # --- build a FairRunSim and keep references ----------------------------------------
        self.run = r.FairRunSim()
        self.run.SetName("TGeant4")
        self.memfile = r.TMemFile("output", "recreate")
        self.sink = r.FairRootFileSink(self.memfile)
        self.run.SetSink(self.sink)
        self.run.SetUserConfig("g4Config_basic.C")  # transport not really used here
        self.rtdb = self.run.GetRuntimeDb()

        self.modules = shipDet_conf.configure(self.run, self.ship_geo)
        self.run.Init()
        print("configured geofile")

        # --- grab geometry objects & keep them ---------------------------------------------
        self.sGeo = self.fgeo.FAIRGeom
        self.top_volume = self.sGeo.GetTopVolume()

        # --- create SciFi mapping ----------------------------------------------------------
        self.SciFiMapping = SciFiMapping.SciFiMapping(self.modules)
        self.SciFiMapping.make_mapping()

        # --- prevent Python from deleting C++ objects --------------------------------------
        for obj in (
            self.run, self.sink, self.memfile,
            self.sGeo, self.top_volume,
            self.SciFiMapping.scifi
        ):
            try:
                r.SetOwnership(obj, False)
            except Exception:
                pass  # ignore for pure-python objects

        # # example position query (can be removed)
        # AF, BF = r.TVector3(), r.TVector3()
        # ch = 101104120
        # self.SciFiMapping.scifi.GetSiPMPosition(ch % 1_000_000, BF, AF)
        # print(f"SiPM position for channel {ch}: {AF.X()}, {AF.Y()}, {AF.Z()}")
        # print(f"SiPM position for channel {ch}: {BF.X()}, {BF.Y()}, {BF.Z()}")
    def _create_chain(self):
        ch = r.TChain("cbmsim")
        for file in self.file_list:
            ch.Add(str(file))
        if ch.GetEntries() == 0:
            print("Error: 'cbmsim' tree not found in the provided files.")
            return None
        return ch

    def get_entries(self):
        return self.chain.GetEntries()

    def get_branches(self):
        return self.chain.GetListOfBranches()

    def analyze_events(self):
        zero_events = 0
        for i, event in enumerate(self.chain):
            if len(event.MtcDetPoint) == 0:
                zero_events += 1
            print(i, len(event.MtcDetPoint))
        print(
            f"Number of events with no hits: {zero_events} out of {self.get_entries()}"
        )



    def store_events_3_histograms(
        self, filename="event_3_histograms.root", plot_type="hist2d"
    ):
        """
        Process the digitized DataFrame and, for each event, build three 2D numpy histograms:
        - Histogram for layer_type == 1 using digitized layer_id and fiber_id_local with fixed binning.
        - Histogram for layer_type == 2 using digitized layer_id and fiber_id_local with fixed binning.
        - Histogram for layer_type == 3 (scintillator) using layer_id and fiber_id_local with specified bins.

        For layer_type 1 and 2, if plot_type is "hist2d", the hit weights are given by the energy loss (Eloss),
        just as is done for layer_type 3. In "scatter" mode the histograms are produced without weights.

        The SciFi-style histograms (for types 1 and 2) are built with:
        - X-axis: digitized layer_id using fixed bins (here we assume 45 layers).
        - Y-axis: digitized fiber_id_local with a number of bins set by self.number_of_fibers.

        The scintillator histogram (for type 3) is built with:
        - X-axis: layer_id (assuming 45 layers).
        - Y-axis: fiber_id_local with 2500 units range (2501 bin edges).

        The histograms (flattened to 1D arrays) and event-related quantities (E_nu, theta_nu, and pdg_nu)
        are stored into a ROOT file (using uproot) as a TTree with branches:
        - event_id, hist_type1, hist_type2, hist_type3, E_nu, theta_nu, and nu_flavor.

        Parameters:
        filename  (str): The ROOT file name to store the histograms.
        plot_type (str): Either "hist2d" (with Eloss weights) or "scatter" (unweighted).
        """
        import numpy as np
        import uproot

        # --- Define fixed binning for layer_type 1 and 2 (SciFi-style) ---
        total_sciFi_layers = 45
        bins_scifi_x = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
        bins_scifi_y = np.linspace(0, self.number_of_fibers, self.number_of_fibers + 1)
        scifi_shape = (len(bins_scifi_x) - 1, len(bins_scifi_y) - 1)

        # --- Define fixed binning for layer_type 3 (Scintillator-style) ---
        bins_scint_x = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
        bins_scint_y = np.linspace(0, 2500, 2501)
        scint_shape = (len(bins_scint_x) - 1, len(bins_scint_y) - 1)

        # --- Prepare lists to hold histogram data for each event ---
        event_ids = []
        hist_type1 = []
        hist_type2 = []
        hist_type3 = []
        E_nu = []
        theta_nu = []
        nu_flavor = []

        # Group the DataFrame by event_id so we process one event at a time.
        grouped = self.df_digi.groupby("event_id")
        for event_id, df_event in grouped:
            event_ids.append(event_id)
            E_nu.append(df_event["E_nu"].iloc[0])
            theta_nu.append(df_event["theta_nu"].iloc[0])
            nu_flavor.append(df_event["pdg_nu"].iloc[0])

            # --- Build histogram for layer_type == 1 ---
            df_type1 = df_event[df_event["layer_type"] == 0]
            if not df_type1.empty:
                if plot_type == "hist2d":
                    H_type1, _, _ = np.histogram2d(
                        df_type1["layer_id"],
                        df_type1["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                        weights=df_type1["Eloss"],
                    )
                elif plot_type == "scatter":
                    H_type1, _, _ = np.histogram2d(
                        df_type1["layer_id"],
                        df_type1["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type1 = np.zeros(scifi_shape)
            hist_type1.append(H_type1.flatten())

            # --- Build histogram for layer_type == 2 ---
            df_type2 = df_event[df_event["layer_type"] == 1]
            if not df_type2.empty:
                if plot_type == "hist2d":
                    H_type2, _, _ = np.histogram2d(
                        df_type2["layer_id"],
                        df_type2["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                        weights=df_type2["Eloss"],
                    )
                elif plot_type == "scatter":
                    H_type2, _, _ = np.histogram2d(
                        df_type2["layer_id"],
                        df_type2["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type2 = np.zeros(scifi_shape)
            hist_type2.append(H_type2.flatten())

            # --- Build histogram for layer_type == 3 ---
            df_type3 = df_event[df_event["layer_type"] == 2]
            if not df_type3.empty:
                if plot_type == "hist2d":
                    H_type3, _, _ = np.histogram2d(
                        df_type3["layer_id"],
                        df_type3["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y],
                        weights=df_type3["Eloss"],
                    )
                elif plot_type == "scatter":
                    H_type3, _, _ = np.histogram2d(
                        df_type3["layer_id"],
                        df_type3["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y],
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type3 = np.zeros(scint_shape)
            hist_type3.append(H_type3.flatten())

        # Convert lists to numpy arrays for storage.
        event_ids = np.array(event_ids, dtype=np.int32)
        hist_type1 = np.array(hist_type1, dtype=np.float32)
        hist_type2 = np.array(hist_type2, dtype=np.float32)
        hist_type3 = np.array(hist_type3, dtype=np.float32)

        # --- Write the histograms and additional variables to a ROOT file ---
        # Create a dictionary to be written as a TTree.
        tree_data = {
            "event_id": event_ids,
            "scifimat_1": hist_type1,
            "scifimat_2": hist_type2,
            "scint": hist_type3,
            "E_nu": E_nu,
            "theta_nu": theta_nu,
            "nu_flavor": nu_flavor,
        }

        # Use uproot to create (or overwrite) the ROOT file with a TTree named "EventTree".
        with uproot.recreate(filename) as root_file:
            root_file["EventTree"] = tree_data

    def read_events_histograms(self, filename="event_histograms.root"):
        """
        Reads the stored ROOT file and recovers the event information.

        Assumes the ROOT file contains a TTree named "EventTree" with the following branches:
        - event_id  : integer event identifier.
        - sciFi_hist: flattened numpy array for the SciFi 2D histogram.
        - scint_hist: flattened numpy array for the scintillator 2D histogram.
        - E_nu      : neutrino energy for the event.
        - nu_flavor : neutrino flavor (as a string or integer representing the PDG code).

        The function reshapes the flattened SciFi histograms into a 2D array with shape:
        (90, self.number_of_fibers)
        and the scintillator histograms into a 2D array with shape:
        (45, 100)

        Returns:
        A dictionary containing:
            - event_id  : numpy array of event IDs.
            - sciFi_hist: numpy array of SciFi histograms, reshaped as (n_events, 90, self.number_of_fibers).
            - scint_hist: numpy array of scintillator histograms, reshaped as (n_events, 45, 100).
            - E_nu      : numpy array of neutrino energies.
            - nu_flavor : numpy array of neutrino flavors.
        """

        # Open the ROOT file and access the TTree.
        with uproot.open(filename) as file:
            tree = file["EventTree"]
            # Retrieve the branches as numpy arrays.
            event_ids = tree["event_id"].array(library="np")
            sciFi_hists_flat = tree["sciFi_hist"].array(library="np")
            scint_hists_flat = tree["scint_hist"].array(library="np")
            E_nu = tree["E_nu"].array(library="np")
            nu_flavor = tree["nu_flavor"].array(library="np")

        # Define the expected shapes of the histograms.
        # SciFi: 90 layers and self.number_of_fibers fibers.
        sciFi_shape = (90, self.number_of_fibers)
        # Scintillator: 45 layers and 100 bins in the fiber/cell direction.
        scint_shape = (45, 2500)

        # Reshape the flattened histograms back to their 2D shapes.
        sciFi_hists = sciFi_hists_flat.reshape(-1, sciFi_shape[0], sciFi_shape[1])
        scint_hists = scint_hists_flat.reshape(-1, scint_shape[0], scint_shape[1])

        # Return the recovered information as a dictionary.
        return {
            "event_id": event_ids,
            "sciFi_hist": sciFi_hists,
            "scint_hist": scint_hists,
            "E_nu": E_nu,
            "nu_flavor": nu_flavor,
        }


    def read_true(self):
        df_digi = {
            "event_id": [],
            "multiplicity": [],
            "track_id": [],
            "E_nu": [],
            "theta_nu": [],
            "px_nu": [],
            "py_nu": [],
            "pz_nu": [],
            "x_nu": [],
            "y_nu": [],
            "z_nu": [],
            "pdg_nu": [],
            "fiber_id": [],
            "fiber_id_local": [],
            "layer_type": [],
            "layer_id": [],
            "x": [],
            "y": [],
            "z": [],
            "pdg": [],
            "Eloss": [],
            "Time": []
        }
        for i, event in enumerate(self.chain):
            E_nu = np.sqrt(
                event.MCTrack[0].GetPx() ** 2
                + event.MCTrack[0].GetPy() ** 2
                + event.MCTrack[0].GetPz() ** 2
            )
            Px_nu, Py_nu, Pz_nu, X_nu, Y_nu, Z_nu = (
                event.MCTrack[0].GetPx(),
                event.MCTrack[0].GetPy(),
                event.MCTrack[0].GetPz(),
                event.MCTrack[0].GetStartX(),
                event.MCTrack[0].GetStartY(),
                event.MCTrack[0].GetStartZ(),
            )
            event_multiplicity = 0
            for mctrack in event.MCTrack:
                if mctrack.GetMotherId() == 0:
                    event_multiplicity += 1

            if len(event.MtcDetPoint) == 0:
                continue

            for hit in event.MtcDetPoint:
                if int(hit.GetDetectorID() / 1e5) % 10 < 2:
                    fiber_id = hit.GetDetectorID()
                    fiber_id_local = hit.GetDetectorID() % 10000
                else:
                    fiber_id = hit.GetDetectorID()
                    fiber_id_local = fiber_id % 10000

                df_digi["event_id"].append(i)
                df_digi["multiplicity"].append(event_multiplicity)
                df_digi["track_id"].append(hit.GetTrackID())
                df_digi["E_nu"].append(E_nu)
                df_digi["theta_nu"].append(np.arccos(Pz_nu / E_nu) * 180 / np.pi)
                df_digi["px_nu"].append(Px_nu)
                df_digi["py_nu"].append(Py_nu)
                df_digi["pz_nu"].append(Pz_nu)
                df_digi["x_nu"].append(X_nu)
                df_digi["y_nu"].append(Y_nu)
                df_digi["z_nu"].append(Z_nu)
                df_digi["pdg_nu"].append(event.MCTrack[0].GetPdgCode())
                df_digi["fiber_id"].append(fiber_id)
                df_digi["layer_type"].append(int(hit.GetDetectorID() / 1e5) % 10)
                df_digi["fiber_id_local"].append(fiber_id_local)
                df_digi["layer_id"].append(int(hit.GetDetectorID() / 1e6) % 100)
                df_digi["x"].append(hit.GetX())
                df_digi["y"].append(hit.GetY())
                df_digi["z"].append(hit.GetZ())
                df_digi["pdg"].append(hit.PdgCode())
                df_digi["Eloss"].append(hit.GetEnergyLoss())
                df_digi["Time"].append(hit.GetTime())
        for key in df_digi:
            print(f"{key}: {len(df_digi[key])}")
        df_digi = pd.DataFrame(df_digi)

        self.df_true = df_digi

    def read_digi(self):
        def cell_xy_from_channel(channel_id: int,
                                nY: int = 50,
                                cell_size: float = 1.0,
                                xmin: float = -25.0,
                                ymin: float = -25.0):
            """Return (x, y) center coordinates of the scint cell."""
            local = channel_id % 10000              # -> i*nY + j
            i, j = divmod(local, nY)               # row=i, col=j
            x = xmin + (i + 0.5) * cell_size
            y = ymin + (j + 0.5) * cell_size
            return x, y
        df_digi = {
            "event_id": [],
            "multiplicity": [],
            "E_nu": [],
            "theta_nu": [],
            "px_nu": [],
            "py_nu": [],
            "pz_nu": [],
            "x_nu": [],
            "y_nu": [],
            "z_nu": [],
            "pdg_nu": [],
            "SiPMID": [],
            "ChanID": [],
            "Signal": [],
            "Time": [],
            "layer_type": [],
            "layer_id": [],
            "local_channel_id": [],
            "x_digi": [],
            "y_digi": [],
            "is_valid": []
        }
        for i, event in enumerate(self.chain):
            E_nu = np.sqrt(
                event.MCTrack[0].GetPx() ** 2
                + event.MCTrack[0].GetPy() ** 2
                + event.MCTrack[0].GetPz() ** 2
            )
            Px_nu, Py_nu, Pz_nu, X_nu, Y_nu, Z_nu = (
                event.MCTrack[0].GetPx(),
                event.MCTrack[0].GetPy(),
                event.MCTrack[0].GetPz(),
                event.MCTrack[0].GetStartX(),
                event.MCTrack[0].GetStartY(),
                event.MCTrack[0].GetStartZ(),
            )
            event_multiplicity = 0
            for mctrack in event.MCTrack:
                if mctrack.GetMotherId() == 0:
                    event_multiplicity += 1

            if len(event.MtcDetPoint) == 0:
                continue
            if not hasattr(event, "Digi_MTCHits"):
                print(f"Event {i} has no Digi_MTCHits, skipping.")
                continue
            for digihit in event.Digi_MTCHits:
                df_digi["event_id"].append(i)
                df_digi["multiplicity"].append(event_multiplicity)
                df_digi["E_nu"].append(E_nu)
                df_digi["theta_nu"].append(np.arccos(Pz_nu / E_nu) * 180 / np.pi)
                df_digi["px_nu"].append(Px_nu)
                df_digi["py_nu"].append(Py_nu)
                df_digi["pz_nu"].append(Pz_nu)
                df_digi["x_nu"].append(X_nu)
                df_digi["y_nu"].append(Y_nu)
                df_digi["z_nu"].append(Z_nu)
                df_digi["pdg_nu"].append(event.MCTrack[0].GetPdgCode())
                df_digi["SiPMID"].append(digihit.GetSiPMChan())
                df_digi["ChanID"].append(digihit.GetChannelID())
                df_digi["Signal"].append(digihit.GetSignal())
                df_digi["Time"].append(digihit.GetTime())
                df_digi["layer_type"].append(digihit.GetStationType())
                df_digi["layer_id"].append(digihit.GetLayer())
                df_digi["local_channel_id"].append(digihit.GetChannelID() % 10000)
                df_digi["is_valid"].append(digihit.isValid())
                if digihit.GetStationType() < 2:
                    AF = r.TVector3()
                    BF = r.TVector3()
                    self.SciFiMapping.scifi.GetSiPMPosition(digihit.GetChannelID() % 1000000, AF, BF)
                    df_digi["x_digi"].append(AF[0])
                    df_digi["y_digi"].append(0.0)
                else:
                    x, y = cell_xy_from_channel(digihit.GetChannelID())
                    df_digi["x_digi"].append(x)
                    df_digi["y_digi"].append(y)
        self.df = pd.DataFrame(df_digi).query("Time < 300 & abs(x_nu) < 20 & abs(y_nu) < 20 & is_valid")

        lc = self.df['local_channel_id'].astype(int)
        sipm_index   = lc // 1000
        sipm_channel = lc % 1000
        self.df['phys_idx'] = sipm_index * 128 + sipm_channel

        # 2) Build the cluster DataFrame
        # self.cluster_df = cluster_scifi_hits(
        #     self.df,
        #     with_qdc=False   # or True if you want Signal-weighting
        # )
        self.cluster_df = cluster_dbscan(
            self.df,
        )
        print(self.cluster_df)
        # exit(0)
        return self.cluster_df


    def study_clustering(self):
        df0 = self.df[self.df.layer_type == 0].copy()
        df1 = self.df[self.df.layer_type == 1].copy()
        df0 = df0.query("event_id < 2 & event_id >= 1")
        df1 = df1.query("event_id < 2 & event_id >= 1")
        df0_cluster = self.cluster_df[self.cluster_df.layer_type == 0].copy()
        df1_cluster = self.cluster_df[self.cluster_df.layer_type == 1].copy()
        df0_cluster = df0_cluster.query("event_id < 2 & event_id >= 1")
        df1_cluster = df1_cluster.query("event_id < 2 & event_id >= 1")
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        binx = np.linspace(0, 45, 46)
        biny = np.linspace(-25, 25, 101)
        ax[0,0].hist2d(df0["layer_id"], df0["x_digi"], bins=(binx, biny), cmap='viridis', cmin=1)
        ax[0,1].hist2d(df1["layer_id"], df1["x_digi"], bins=(binx, biny), cmap='viridis', cmin=1)
        ax[0,0].set_xlabel("Layer ID")
        ax[0,0].set_ylabel("x_digi")
        ax[0,1].set_xlabel("Layer ID")
        ax[0,1].set_ylabel("x_digi")
        ax[0,0].set_title("Layer 0 (SciFi)")
        ax[0,1].set_title("Layer 1 (SciFi)")
        ax[1,0].hist2d(df0_cluster["layer_id"], df0_cluster["x_digi"], bins=(binx, biny), cmap='viridis', cmin=1)
        ax[1,1].hist2d(df1_cluster["layer_id"], df1_cluster["x_digi"], bins=(binx, biny), cmap='viridis', cmin=1)
        ax[1,0].set_xlabel("Layer ID")
        ax[1,0].set_ylabel("x_digi")
        ax[1,1].set_xlabel("Layer ID")
        ax[1,1].set_ylabel("x_digi")
        ax[1,0].set_title("Layer 0 (SciFi). Clustering")
        ax[1,1].set_title("Layer 1 (SciFi). Clustering")
        plt.tight_layout()
        fig.savefig("figures/clustering.pdf")




    def study_clustering_3d(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        # Filter original DataFrame for layers 0 and 1, and event_id in [1, 2)
        df0 = self.df[(self.df.layer_type == 0) & (self.df.event_id >= 1) & (self.df.event_id < 2)]
        df1 = self.df[(self.df.layer_type == 1) & (self.df.event_id >= 1) & (self.df.event_id < 2)]

        # Filter clustered DataFrame similarly
        df0_cluster = self.cluster_df[(self.cluster_df.layer_type == 0) & (self.cluster_df.event_id >= 1) & (self.cluster_df.event_id < 2)]
        df1_cluster = self.cluster_df[(self.cluster_df.layer_type == 1) & (self.cluster_df.event_id >= 1) & (self.cluster_df.event_id < 2)]

        # Create a 2x2 grid of 3D scatter plots
        fig = plt.figure(figsize=(14, 12))

        # Plot for original layer 0
        ax0 = fig.add_subplot(2, 2, 1, projection='3d')
        p0 = ax0.scatter(
            df0['layer_id'],
            df0['x_digi'],
            df0['Time'],
            c=df0['Time'],
            cmap='viridis',
            marker='o',
            alpha=0.7
        )
        ax0.set_title('Layer 0 (SciFi)')
        ax0.set_xlabel('Layer ID')
        ax0.set_ylabel('x_digi')
        ax0.set_zlabel('Time')
        fig.colorbar(p0, ax=ax0, pad=0.1, label='Time')

        # Plot for original layer 1
        ax1 = fig.add_subplot(2, 2, 2, projection='3d')
        p1 = ax1.scatter(
            df1['layer_id'],
            df1['x_digi'],
            df1['Time'],
            c=df1['Time'],
            cmap='viridis',
            marker='o',
            alpha=0.7
        )
        ax1.set_title('Layer 1 (SciFi)')
        ax1.set_xlabel('Layer ID')
        ax1.set_ylabel('x_digi')
        ax1.set_zlabel('Time')
        fig.colorbar(p1, ax=ax1, pad=0.1, label='Time')

        # Plot for clustered layer 0
        ax2 = fig.add_subplot(2, 2, 3, projection='3d')
        p2 = ax2.scatter(
            df0_cluster['layer_id'],
            df0_cluster['x_digi'],
            df0_cluster['Time'],
            c=df0_cluster['Time'],
            cmap='plasma',
            marker='^',
            alpha=0.7
        )
        ax2.set_title('Layer 0 (SciFi) Clustering')
        ax2.set_xlabel('Layer ID')
        ax2.set_ylabel('x_digi')
        ax2.set_zlabel('Time')
        fig.colorbar(p2, ax=ax2, pad=0.1, label='Time')

        # Plot for clustered layer 1
        ax3 = fig.add_subplot(2, 2, 4, projection='3d')
        p3 = ax3.scatter(
            df1_cluster['layer_id'],
            df1_cluster['x_digi'],
            df1_cluster['Time'],
            c=df1_cluster['Time'],
            cmap='plasma',
            marker='^',
            alpha=0.7
        )
        ax3.set_title('Layer 1 (SciFi) Clustering')
        ax3.set_xlabel('Layer ID')
        ax3.set_ylabel('x_digi')
        ax3.set_zlabel('Time')
        fig.colorbar(p3, ax=ax3, pad=0.1, label='Time')

        plt.tight_layout()
        fig.savefig('figures/clustering_3d.pdf', dpi=300)
        plt.show()

    def study_clustering_2d(self):
        # Filter original DataFrame for layers 0 and 1, and event_id in [1, 2)
        df0 = self.df[(self.df.layer_type == 0) & (self.df.event_id >= 1) & (self.df.event_id < 2)]
        df1 = self.df[(self.df.layer_type == 1) & (self.df.event_id >= 1) & (self.df.event_id < 2)]

        # Filter clustered DataFrame similarly
        df0_cluster = self.cluster_df[(self.cluster_df.layer_type == 0) & (self.cluster_df.event_id >= 1) & (self.cluster_df.event_id < 2)]
        df1_cluster = self.cluster_df[(self.cluster_df.layer_type == 1) & (self.cluster_df.event_id >= 1) & (self.cluster_df.event_id < 2)]

        # Create a 2x2 grid of 2D scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Original layer 0
        sc0 = axes[0, 0].scatter(
            df0['layer_id'],
            df0['x_digi'],
            c=df0['Time'],
            cmap='viridis',
            alpha=0.7,
            label = f'{len(df0)} hits'
        )
        axes[0, 0].set_title('Layer 0 (SciFi)')
        axes[0, 0].set_xlabel('Layer ID')
        axes[0, 0].set_ylabel('x_digi')
        axes[0, 0].legend()
        fig.colorbar(sc0, ax=axes[0, 0], pad=0.05, label='Time')

        # Original layer 1
        sc1 = axes[0, 1].scatter(
            df1['layer_id'],
            df1['x_digi'],
            c=df1['Time'],
            cmap='viridis',
            alpha=0.7,
            label = f'{len(df1)} hits'
        )
        axes[0, 1].set_title('Layer 1 (SciFi)')
        axes[0, 1].set_xlabel('Layer ID')
        axes[0, 1].set_ylabel('x_digi')
        axes[0, 1].legend()
        fig.colorbar(sc1, ax=axes[0, 1], pad=0.05, label='Time')

        # Clustered layer 0
        sc2 = axes[1, 0].scatter(
            df0_cluster['layer_id'],
            df0_cluster['x_digi'],
            c=df0_cluster['Time'],
            cmap='plasma',
            alpha=0.7,
            label = f'{len(df0_cluster)} clusters'
        )
        axes[1, 0].set_title('Layer 0 (SciFi) Clustering')
        axes[1, 0].set_xlabel('Layer ID')
        axes[1, 0].set_ylabel('x_digi')
        axes[1, 0].legend()
        fig.colorbar(sc2, ax=axes[1, 0], pad=0.05, label='Time')

        # Clustered layer 1
        sc3 = axes[1, 1].scatter(
            df1_cluster['layer_id'],
            df1_cluster['x_digi'],
            c=df1_cluster['Time'],
            cmap='plasma',
            alpha=0.7,
            label = f'{len(df1_cluster)} clusters'
        )
        axes[1, 1].set_title('Layer 1 (SciFi) Clustering')
        axes[1, 1].set_xlabel('Layer ID')
        axes[1, 1].set_ylabel('x_digi')
        axes[1, 1].legend()
        fig.colorbar(sc3, ax=axes[1, 1], pad=0.05, label='Time')

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig('figures/clustering_2d.pdf', dpi=300)
        plt.show()


    def plot_clustering_efficiency(self, event_id=1):
        # restrict to one event and SciFi layers
        hits   = self.df.query("event_id==@event_id & layer_type<2")
        clusts = self.cluster_df.query("event_id==@event_id & layer_type<2")

        # count per layer
        hit_counts   = hits.groupby('layer_id').size()
        clus_counts  = clusts.groupby('layer_id').size()

        # ensure both share the same set of layer_ids
        layers = np.arange(hits['layer_id'].min(), hits['layer_id'].max()+1)
        H = hit_counts.reindex(layers, fill_value=0)
        C = clus_counts.reindex(layers, fill_value=0)

        # bar positions
        width = 0.4
        x = layers

        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.bar(x - width/2, H, width, label='Raw hits')
        ax1.bar(x + width/2, C, width, label='Clusters')
        ax1.set_xlabel('Layer ID')
        ax1.set_ylabel('Count')
        ax1.legend(loc='upper left')

        # efficiency on a second y-axis
        eff = C / H.replace(0, np.nan)
        ax2 = ax1.twinx()
        # disable its grid:
        ax2.grid(False)

        # make its background transparent so you see only ax1’s grid
        ax2.patch.set_alpha(0)
        ax2.plot(x, 1 - eff, marker='o', linestyle='-', color='tab:gray', label='Clustering efficiency')
        ax2.set_ylabel('1 - #clusters / #hits')
        ax2.set_ylim(0,1.05)
        ax2.legend(loc='upper right')

        plt.title(f'Event {event_id}: Hit vs Cluster counts by Layer')
        plt.tight_layout()
        fig.savefig(f"figures/clustering_efficiency_event_{event_id}.pdf")

    def reconstruct_scifi_digihit_position(self,
                        dt_shift=0.235/30,   # nanoseconds (adjust units!)
                        tol=1.,                # allowed |Δt - dt_shift|, same units as Time
                        one_to_one=True,
                        clustering = False):
        """
        Return a dataframe of matched (layer0, layer1) digi hits.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: event_id, layer_type (0/1), layer_id, Time (numeric).
        dt_shift : float
            Expected time difference L1-L0. Make sure it's in the same units as df['Time'].
        tol : float
            Matching tolerance around dt_shift.
        one_to_one : bool
            If True, each L1 hit will be paired at most once (nearest). Otherwise many-to-many pairs are kept.
        """
        print("Reconstructing SciFi DigiHit positions...")
        print(self.df.columns)
        print(self.df.query("layer_type == 2")[["x_digi", "y_digi", "ChanID"]].head())

        # # Split
        # df0 = self.df[self.df.layer_type == 0].copy()
        # df1 = self.df[self.df.layer_type == 1].copy()
        # df0 = df0.query("event_id < 100")
        # df1 = df1.query("event_id < 100")
        # # Shift L1 times so we can merge on equality
        # df1['Time_corr'] = df1['Time'] - dt_shift

        # Sort for merge_asof
        # Split
        if not clustering:
            df0 = self.df[self.df.layer_type == 0].copy()
            df1 = self.df[self.df.layer_type == 1].copy()
        else:
            df0 = self.cluster_df[self.cluster_df.layer_type == 0].copy()
            df1 = self.cluster_df[self.cluster_df.layer_type == 1].copy()
            # df0.rename(columns={"mean_x": "x_digi", "mean_y": "y_digi"}, inplace=True)
            # df1.rename(columns={"mean_x": "x_digi", "mean_y": "y_digi"}, inplace=True)
        # df0 = df0.query("event_id < 235 & event_id >= 225")
        # df1 = df1.query("event_id < 235 & event_id >= 225")
        df0 = df0.query("event_id < 10 & event_id >= 1")
        df1 = df1.query("event_id < 10 & event_id >= 1")
        # df0 = df0.query("event_id < 160 & event_id >= 140")
        # df1 = df1.query("event_id < 160 & event_id >= 140")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].hist(df0["x_digi"], bins=100, histtype='step', label="L0 x_digi")
        # ax[1].hist(df1["x_digi"], bins=100, histtype='step', label="L1 x_digi")
        # ax[0].set_xlabel("x_digi")
        # ax[0].set_ylabel("Counts")
        # ax[1].set_xlabel("x_digi")
        # ax[1].set_ylabel("Counts")
        # ax[0].legend()
        # ax[1].legend()
        binx = np.linspace(0, 45, 46)
        biny = np.linspace(-25, 25, 101)
        ax[0].hist2d(df0["layer_id"], df0["x_digi"], bins=(binx, biny), cmap='viridis', cmin=1)
        ax[1].hist2d(df1["layer_id"], df1["x_digi"], bins=(binx, biny), cmap='viridis', cmin=1)
        ax[0].set_xlabel("Layer ID")
        ax[0].set_ylabel("x_digi")
        ax[1].set_xlabel("Layer ID")
        ax[1].set_ylabel("x_digi")
        ax[0].set_title("Layer 0 (SciFi)")
        ax[1].set_title("Layer 1 (SciFi)")
        plt.tight_layout()
        fig.savefig("figures/sciFi_digi_x_digi_clustering.pdf")

        # Shift L1 times so we can merge on equality
        df1['Time_corr'] = df1['Time'] - dt_shift


        fig, ax = plt.subplots(figsize = (8, 6))
        bins = np.linspace(0, 20, 101)
        ax.hist(df0["Time"], bins=bins, histtype='step', label="SciFi plane U")
        ax.hist(df1["Time_corr"], bins=bins, histtype='step', label="SciFi plane V (shifted)")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Counts")
        ax.legend()
        plt.tight_layout()
        fig.savefig("figures/sciFi_digi_time.pdf")

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].hist2d(df0["layer_id"], df0["Time"], bins=(binx, bins), cmap='viridis', cmin=1)
        ax[1].hist2d(df1["layer_id"], df1["Time_corr"], bins=(binx, bins), cmap='viridis', cmin=1)
        ax[0].set_xlabel("Layer ID")
        ax[0].set_ylabel("Time [ns]")
        ax[1].set_xlabel("Layer ID")
        ax[1].set_ylabel("Time [ns]")
        ax[0].set_title("Layer 0 (SciFi)")
        ax[1].set_title("Layer 1 (SciFi)")
        plt.tight_layout()
        fig.savefig("figures/sciFi_digi_time_layer.pdf")

        # # --- ADD THIS BLOCK ---------------------------------
        # # Time is in ns; create a strictly increasing key per (event_id, layer_id)
        # max_t   = max(df0['Time'].max(), df1['Time_corr'].max()) + 1.0   # span > any Time
        # grp0    = df0['event_id'] * 1000 + df0['layer_id']               # unique group id
        # grp1    = df1['event_id'] * 1000 + df1['layer_id']
        # df0['_tkey'] = df0['Time']      + grp0 * max_t
        # df1['_tkey'] = df1['Time_corr'] + grp1 * max_t

        # df0 = df0.sort_values('_tkey', kind='mergesort').reset_index(drop=True)
        # df1 = df1.sort_values('_tkey', kind='mergesort').reset_index(drop=True)
        # # -----------------------------------------------------
        # chk = df0.groupby(['event_id','layer_id'])['Time'].apply(lambda s: s.is_monotonic_increasing).all()
        # print("Left sorted per group:", chk)
        # print("!!! first plane: ", df0[["event_id", "Time", "layer_id", "_tkey"]].head())
        # print("!!! second plane: ", df1[["event_id", "Time_corr", "layer_id", "_tkey"]].head())
        dx_max = 100*np.tan(np.radians(5))  # your x-window

        if one_to_one:
            # one-to-one: for each L0 hit, only consider L1 hits within the x-window,
            # then pick the time-closest among those (and apply tol).
            matched = []
            # breakpoint()
            for (ev, lid), g0_grp in df0.groupby(['event_id','layer_id']):
                # if lid == 7:
                #     breakpoint()
                g1_grp = df1[(df1.event_id == ev) & (df1.layer_id == lid)]
                print(f"Processing event {ev}, layer {lid}: L0 hits {len(g0_grp)}, L1 hits {len(g1_grp)}")
                if g1_grp.empty:
                    continue
                for _, r0 in g0_grp.iterrows():
                    # 1) filter by x window:
                    cand = g1_grp[
                        (r0.x_digi < g1_grp.x_digi) &
                        (np.abs(r0.x_digi - g1_grp.x_digi) < dx_max)
                    ]
                    if cand.empty:
                        continue
                    # 2) compute time difference and apply tol
                    cand = cand.copy()
                    cand['dt_diff'] = (cand.Time_corr - r0.Time).abs()
                    cand = cand[cand.dt_diff <= tol]
                    if cand.empty:
                        continue
                    # 3) pick the best (smallest dt_diff)
                    best_idx = cand['dt_diff'].idxmin()
                    best = cand.loc[best_idx]

                    # 4) exclude the chosen candidate from future consideration
                    g1_grp = g1_grp.drop(index=best_idx)


                    # 5) merge into one record with _L0/_L1 suffixes
                    rec = {f"{k}_L0": r0[k] for k in df0.columns}
                    rec.update({f"{k}_L1": best[k] for k in df1.columns})
                    rec.update({
                        'dt_diff': best['dt_diff']})

        # # --- 4) geometric intersection
        #             x_u = r0.x_digi
        #             x_v = best.x_digi
        #             # solve:  U-line: (x,y) = (x_u,25) + t*(cos5°, –sin5°)
        #             #         V-line: (x,y) = (x_v,25) + t*(–cos5°, –sin5°)
        #             t = (x_v - x_u) / (2*np.cos(np.radians(5)))
        #             x_int = x_u + np.cos(np.radians(5)) * t
        #             y_int = 25  - np.sin(np.radians(5)) * t

        #             rec = {f"{k}_L0": r0[k] for k in df0.columns}
        #             rec.update({f"{k}_L1": best[k] for k in df1.columns})
        #             rec['real_x_digi'] = x_int
        #             rec['real_y_digi'] = y_int
                    matched.append(rec)

            self.pairs = pd.DataFrame(matched)

        else:
            # many-to-many: cross-join within each (event,layer), then apply both cuts
            g0 = df0.copy()
            g1 = df1.copy()
            # time window columns
            g0['tmin'] = g0['Time'] - tol
            g0['tmax'] = g0['Time'] + tol

            all_pairs = (
                g0
                .merge(g1, on=['event_id','layer_id'], suffixes=('_L0','_L1'))
            )
            mask_time = (
                (all_pairs.Time_corr >= all_pairs.tmin) &
                (all_pairs.Time_corr <= all_pairs.tmax)
            )
            mask_dx = (
                (all_pairs.x_digi_L0 >  all_pairs.x_digi_L1) &
                (all_pairs.x_digi_L0 <  all_pairs.x_digi_L1 + dx_max)
            )

            filt = all_pairs[mask_time & mask_dx]
            self.pairs = filt.drop(columns=['tmin','tmax']).reset_index(drop=True)

        print(self.pairs[["event_id_L0", "layer_id_L0", "x_digi_L0", "x_digi_L1", "Time_L0", "Time_L1"]])
        fig, ax = plt.subplots(1, 3, figsize = (18, 6))
        ax[0].hist(self.pairs["dt_diff"], bins=100, histtype='step', label="dt_diff (L1 - L0)")
        ax[0].set_xlabel("Time difference (L1 - L0) [ns]")
        ax[0].set_ylabel("Counts")
        ax[0].set_title("Time difference between L1 and L0 DigiHits")
        ax[0].legend()
        ax[1].hist2d(self.pairs["layer_id_L0"], self.pairs["dt_diff"], bins=(binx, 100), cmap='viridis', cmin=1)
        ax[1].set_xlabel("Layer ID")
        ax[1].set_ylabel("Time difference (L1 - L0) [ns]")
        ax[1].set_title("Time difference between L1 and L0 DigiHits per Layer ID")
        ax[2].hist2d(self.pairs["dt_diff"], 25 - (self.pairs["x_digi_L1"] - self.pairs["x_digi_L0"]) / (2 * np.tan(np.radians(5))), bins=(100, 100), cmap='viridis', cmin=1)
        ax[2].set_xlabel("Time difference (L1 - L0) [ns]")
        ax[2].set_ylabel("Reconstructed y position [cm]")
        ax[2].set_title("Reconstructed y position vs Time difference")
        plt.tight_layout()
        fig.savefig("figures/sciFi_digi_time_difference.pdf")
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].hist((self.pairs["x_digi_L1"] + self.pairs["x_digi_L0"]) / 2, bins=100, histtype='step', label='L1 - L0')
        ax[0].set_xlabel('(x_digi_L1 + x_digi_L0) / 2')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('x_digi_real')
        ax[1].hist(self.pairs["x_digi_L1"] - self.pairs["x_digi_L0"], bins=100, histtype='step', label='L1 - L0')
        ax[1].set_xlabel('x_digi_L1 - x_digi_L0 1')
        ax[1].set_ylabel('Counts')
        ax[1].set_title('Difference in x_digi between L1 and L0')
        fig.savefig("figures/sciFi_digi_hit_position_reconstruction.pdf")
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        # Calculate reconstructed positions
        self.pairs["real_x_digi"] = (self.pairs["x_digi_L1"] + self.pairs["x_digi_L0"]) / 2
        print((self.pairs["x_digi_L0"] - self.pairs["x_digi_L1"]) / 2, np.tan(np.radians(85)))
        self.pairs["real_y_digi"] = (25 - (self.pairs["x_digi_L1"] - self.pairs["x_digi_L0"]) / (np.tan(np.radians(5)) * 2.)) / 1
        # 2D histogram for (real_x_digi, real_y_digi)
        h = ax[0].hist2d(
            self.pairs["real_x_digi"],
            self.pairs["real_y_digi"],
            bins=(100, 100),  # 50 cm range, 2.5 mm bins
            norm=LogNorm(),
            cmap="viridis"
        )
        fig.colorbar(h[3], ax=ax[0], label="Counts (log scale)")
        ax[0].set_xlabel('Reconstructed x [cm]')
        ax[0].set_ylabel('Reconstructed y [cm]')
        ax[0].set_title('2D Reconstructed SciFi DigiHit Positions')
        # 2D histogram for (x_digi_L0, x_digi_L1)
        h2 = ax[1].hist2d(
            self.pairs["x_digi_L0"],
            self.pairs["x_digi_L1"],
            bins=100,
            norm=LogNorm(),
            cmap="viridis"
        )
        fig.colorbar(h2[3], ax=ax[1], label="Counts (log scale)")
        ax[1].set_xlabel('x_digi_L0 [cm]')
        ax[1].set_ylabel('x_digi_L1 [cm]')
        ax[1].set_title('x_digi_L0 vs x_digi_L1')
        fig.tight_layout()
        fig.savefig("figures/sciFi_digi_hit_position_reconstruction_2d_1.pdf")

    def merge_back_scint(self):
        # 1) rename to the canonical names
        pairs_clean = self.pairs.rename(columns={
            'real_x_digi': 'x_digi',
            'real_y_digi': 'y_digi',
            'layer_id_L0': 'layer_id',
            'layer_type_L0': 'layer_type',
            'event_id_L0': 'event_id',
            "E_nu_L0": "E_nu",
            "pdg_nu_L0": "pdg_nu",
        }).copy()
        pairs_clean["Signal"] = (pairs_clean["Signal_L0"] + pairs_clean["Signal_L1"]) / 2.
        pairs_clean["Time"] = (pairs_clean["Time_L0"] + pairs_clean["Time_L1"]) / 2.
        # 2) keep only the rows that were NOT part of the pairing (layer_type == 2)
        df2 = self.df[self.df.layer_type == 2].copy()

        # 3) unify columns and concat
        final_cols = df2.columns.union(pairs_clean.columns)
        self.df_merged = pd.concat(
            [pairs_clean.reindex(columns=final_cols),
            df2.reindex(columns=final_cols)],
            ignore_index=True,
            sort=False
        )

        print("Merged dataframe shape:", self.df_merged[["event_id", "layer_type", "x_digi", "y_digi", "layer_id"]])
        print("Merged dataframes:", self.df_merged.loc[(self.df_merged.event_id.isin([1,2,3])) & (self.df_merged.layer_type.isin([0,1]))][["event_id", "layer_type", "x_digi", "y_digi", "layer_id"]])



    def vis_digi_1(self,
                event=0,
                merged=True,
                x_range=(-25, 25),
                y_range=(-25, 25),
                nbins_xy=50,
                log_scale=True,
                outfile="figures/event_display_planes_{event}.pdf",
                make_3d=False,
                outfile3d="figures/event_display_3d_{event}.pdf",
                point_scale=50,
                point_alpha=0.8,
                two_by_two=False):
        """
        3×2 event display (default) OR:
        - 3×1 3D scatter plots if make_3d=True
        - 2×2 compact view if two_by_two=True:
            Row 0: SciFi (layer_type 0 & 1)  ->  y–z  |  x–z
            Row 1: Scint (layer_type 2)     ->  y–z  |  x–z

        Assumes df has: x_digi, y_digi, layer_id, layer_type, Signal, pdg_nu, E_nu, event_id.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, SymLogNorm

        # pick the event
        if merged:
            df_evt = self.df_merged[self.df_merged['event_id'] == event].copy()
        else:
            df_evt = self.df[self.df['event_id'] == event].copy()
        if df_evt.empty:
            print(f"No entries for event {event}")
            return

        # neutrino info
        pdg_map = {12: 'nu_e', -12: 'anu_e', 14: 'nu_mu', -14: 'anu_mu', 16: 'nu_tau', -16: 'nu_tau'}
        pdg = df_evt['pdg_nu'].iloc[0]
        nu_type = pdg_map.get(pdg, f'pdg_{pdg}')
        E = df_evt['E_nu'].iloc[0]

        # common bins / ranges
        z_min = 0
        z_max = 45
        bin_z = np.arange(z_min - 0.5, z_max + 1.5, 1.0)
        bin_x = np.linspace(x_range[0], x_range[1], nbins_xy + 1)
        bin_y = np.linspace(y_range[0], y_range[1], nbins_xy + 1)
        norm = LogNorm() if log_scale else None

        layers = [0, 1, 2]

        if not make_3d and two_by_two:
            print("Using 2x2 compact mode for visualization.")
            # ----------------- 2×2 COMPACT MODE -----------------
            scifi_hits  = df_evt[df_evt['layer_type'].isin([0, 1])]
            scint_hits  = df_evt[df_evt['layer_type'] == 2]

            fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex='col')

            # def plot_plane(ax, z_vals, y_or_x_vals, bins_zy, weights, ylabel, title):
            #     print("Plotting plane with z_vals:", z_vals[:5], "y_or_x_vals:", y_or_x_vals[:5], "weights:", weights[:5])
            #     h = ax.hist2d(z_vals, y_or_x_vals,
            #                 bins=bins_zy,
            #                 # weights=weights,
            #                 norm=norm,
            #                 cmap="viridis")
            #     ax.set_ylabel(ylabel)
            #     ax.set_title(title)
            #     fig.colorbar(h[3], ax=ax)
            def plot_plane(ax, x_vals, y_or_x_vals, bins_zy, time_vals, ylabel, title):
                """
                Scatter plot in 2D with Time mapped to color.
                Parameters:
                - ax: matplotlib Axes
                - x_vals: array-like → one axis (e.g. layer_id or x_digi)
                - y_or_x_vals: array-like → the other axis (x_digi or layer_id)
                - time_vals: array-like → values for coloring each point by Time
                - xlabel, ylabel: strings for axis labels
                - title: string for the subplot title
                """
                sc = ax.scatter(
                    x_vals,
                    y_or_x_vals,
                    c=time_vals,
                    cmap='viridis',
                    alpha=0.7,
                    # edgecolor='',
                )
                ax.set_ylabel(ylabel)
                ax.set_title(title)

            # Row 0: SciFi
            if scifi_hits.empty:
                for c in range(2):
                    axes[0, c].text(0.5, 0.5, "No SciFi hits", ha='center', va='center',
                                    transform=axes[0, c].transAxes)
                    axes[0, c].set_axis_off()
            else:
                print("scifi_hits: ", scifi_hits.head())
                z_vals = scifi_hits['layer_id'].to_numpy()
                x_vals = scifi_hits['x_digi'].to_numpy()
                y_vals = scifi_hits['y_digi'].to_numpy()
                # w_vals = scifi_hits['Signal'].to_numpy()
                # w_vals += np.abs(w_vals.min()) + 1e-6  # avoid log(0) issues
                # print(w_vals[w_vals <= 0])
                w_vals = scifi_hits["Time"].to_numpy()  # use Time for color mapping
                bin_x = np.linspace(x_range[0], x_range[1], 100 + 1)
                bin_y = np.linspace(y_range[0], y_range[1], 100 + 1)
                print(f"z_vals: {z_vals[:5]}, x_vals: {x_vals[:5]}, y_vals: {y_vals[:5]}, w_vals: {w_vals[:5]}")
                plot_plane(axes[0, 0], z_vals, y_vals, (bin_z, bin_y), w_vals,
                        'y_digi', 'SciFi: y–z')
                plot_plane(axes[0, 1], z_vals, x_vals, (bin_z, bin_x), w_vals,
                        'x_digi', 'SciFi: x–z')

            # Row 1: Scint
            if scint_hits.empty:
                for c in range(2):
                    axes[1, c].text(0.5, 0.5, "No Scint hits", ha='center', va='center',
                                    transform=axes[1, c].transAxes)
                    axes[1, c].set_axis_off()
            else:
                z_vals = scint_hits['layer_id'].to_numpy()
                x_vals = scint_hits['x_digi'].to_numpy()
                y_vals = scint_hits['y_digi'].to_numpy()
                # w_vals = scint_hits['Signal'].to_numpy()
                w_vals = scint_hits["Time"].to_numpy()  # use Time for color mapping
                bin_x = np.linspace(x_range[0], x_range[1], nbins_xy + 1)
                bin_y = np.linspace(y_range[0], y_range[1], nbins_xy + 1)
                plot_plane(axes[1, 0], z_vals, y_vals, (bin_z, bin_y), w_vals,
                        'y_digi', 'Scint: y–z')
                plot_plane(axes[1, 1], z_vals, x_vals, (bin_z, bin_x), w_vals,
                        'x_digi', 'Scint: x–z')

            axes[1, 0].set_xlabel('z (layer_id)')
            axes[1, 1].set_xlabel('z (layer_id)')

            fig.suptitle(f'{nu_type}, E = {E:.2f} GeV (event {event})', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            outpath = outfile.format(event=event)
            outfile = outfile.replace(".pdf", f"_{event}_{'merged' if merged else ''}2x2.pdf")
            fig.savefig(outpath)
            print(f"Saved: {outpath}")

        elif not make_3d:
            # ----------------- 3×2 HIST MODE (your original) -----------------
            fig, axes = plt.subplots(3, 2, figsize=(18, 22), sharex='col')

            for row, layer in enumerate(layers):
                hits = df_evt[df_evt['layer_type'] == layer]
                if hits.empty:
                    for col in range(2):
                        ax = axes[row, col]
                        ax.text(0.5, 0.5, f"No hits (layer {layer})",
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_axis_off()
                    continue

                z_vals = hits['layer_id'].to_numpy()
                x_vals = hits['x_digi'].to_numpy()
                y_vals = hits['y_digi'].to_numpy()
                w_vals = hits['Signal'].to_numpy()

                # Left: x_digi vs z
                h1 = axes[row, 0].hist2d(z_vals, x_vals,
                                        bins=(bin_z, bin_x),
                                        weights=w_vals,
                                        norm=norm,
                                        cmap="viridis")
                axes[row, 0].set_ylabel('x_digi')
                axes[row, 0].set_title(f'Layer {layer}: x–z')
                fig.colorbar(h1[3], ax=axes[row, 0])

                # Right: y_digi vs z
                h2 = axes[row, 1].hist2d(z_vals, y_vals,
                                        bins=(bin_z, bin_y),
                                        weights=w_vals,
                                        norm=norm,
                                        cmap="viridis")
                axes[row, 1].set_ylabel('y_digi')
                axes[row, 1].set_title(f'Layer {layer}: y–z')
                fig.colorbar(h2[3], ax=axes[row, 1])

            axes[2, 0].set_xlabel('z (layer_id)')
            axes[2, 1].set_xlabel('z (layer_id)')

            fig.suptitle(f'{nu_type}, E = {E:.2f} GeV (event {event})', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            outpath = outfile.format(event=event)
            outfile = outfile.replace(".pdf", f"_{event}_{merged}.pdf")
            fig.savefig(outpath)
            print(f"Saved: {outpath}")

        else:
            # ----------------- 3D SCATTER MODE -----------------
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(22, 18))
            for idx, layer in enumerate(layers):
                ax = fig.add_subplot(3, 1, idx + 1, projection='3d')
                hits = df_evt[df_evt['layer_type'] == layer]
                if hits.empty:
                    ax.text2D(0.5, 0.5, f"No hits (layer {layer})",
                            transform=ax.transAxes, ha='center', va='center')
                    ax.set_axis_off()
                    continue

                x_vals = hits['x_digi'].to_numpy()
                y_vals = hits['y_digi'].to_numpy()
                z_vals = hits['layer_id'].to_numpy()
                w_vals = hits['Signal'].to_numpy()

                if w_vals.max() > 0:
                    sizes = point_scale * (w_vals / w_vals.max())
                else:
                    sizes = np.full_like(w_vals, point_scale * 0.2)

                sc = ax.scatter(x_vals, y_vals, z_vals,
                                c=w_vals,
                                s=sizes,
                                alpha=point_alpha,
                                cmap='viridis',
                                norm=norm)

                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_zlim(z_min, z_max)

                ax.set_xlabel('x_digi')
                ax.set_ylabel('y_digi')
                ax.set_zlabel('z (layer_id)')
                ax.set_title(f'Layer {layer}: x_digi–y_digi–z')

                cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.6)
                cb.set_label('Signal')

            fig.suptitle(f'{nu_type}, E = {E:.2f} GeV (event {event})', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            outpath3d = outfile3d.format(event=event)
            outfile3d = outfile3d.replace(".pdf", f"_{event}_{'merged' if merged else ''}.pdf")
            fig.savefig(outpath3d)
            print(f"Saved: {outpath3d}")


    def vis_true(self,
                event=0,
                mode="2x2",
                x_range=(-25, 25),
                y_range=(-25, 25),
                nbins=50,
                outfile2x2="figures/true_display_2x2_{event}.pdf",
                outfile3d="figures/true_display_3d_{event}.pdf"):
        """
        Visualize true hits from self.df_true for a given event.

        Parameters
        ----------
        event : int
            Event ID to display.
        mode : str, {"2x2", "3d"}
            - "2x2": 2×2 histogram view:
                Row 0: SciFi (layer_type 0 & 1) → y–z | x–z
                Row 1: Scint (layer_type 2)    → y–z | x–z
            - "3d": two 3D scatter plots (SciFi & Scint).
        x_range, y_range : tuple[float, float]
            Bounds for x and y histograms / scatter axes.
        nbins : int
            Number of bins in each dimension for histograms.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # select event
        df_evt = self.df_true[self.df_true['event_id'] == event]
        if df_evt.empty:
            print(f"No entries for event {event} in df_true")
            return

        # split SciFi vs Scint
        scifi = df_evt[df_evt['layer_type'].isin([0, 1])]
        scint = df_evt[df_evt['layer_type'] == 2]

        # common bins
        z_min, z_max = 0, 45
        bin_z = np.arange(z_min - 0.5, z_max + 1.5, 1.0)
        bin_x = np.linspace(x_range[0], x_range[1], nbins + 1)
        bin_y = np.linspace(y_range[0], y_range[1], nbins + 1)
        norm = LogNorm()

        if mode == "2x2":
            fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex='col')
            # def _hist(ax, z, v, bins, weights, xlabel, title):
            #     h = ax.hist2d(z, v, bins=bins, weights=weights, norm=norm, cmap="viridis")
            #     ax.set_ylabel(xlabel)
            #     ax.set_title(title)
            #     fig.colorbar(h[3], ax=ax)

            def _hist(ax, x_vals, y_or_x_vals, bins_zy, time_vals, ylabel, title):
                """
                Scatter plot in 2D with Time mapped to color.
                Parameters:
                - ax: matplotlib Axes
                - x_vals: array-like → one axis (e.g. layer_id or x_digi)
                - y_or_x_vals: array-like → the other axis (x_digi or layer_id)
                - time_vals: array-like → values for coloring each point by Time
                - xlabel, ylabel: strings for axis labels
                - title: string for the subplot title
                """
                sc = ax.scatter(
                    x_vals,
                    y_or_x_vals,
                    c=time_vals,
                    cmap='viridis',
                    alpha=0.7,
                    # edgecolor='',
                )
                ax.set_ylabel(ylabel)
                ax.set_title(title)

            # Row 0: SciFi
            if scifi.empty:
                for c in (0, 1):
                    axes[0, c].text(0.5, 0.5, "No SciFi hits", ha='center', va='center', transform=axes[0, c].transAxes)
                    axes[0, c].set_axis_off()
            else:
                z = scifi['layer_id'].to_numpy()
                x = scifi['x'].to_numpy()
                y = scifi['y'].to_numpy()
                # w = scifi['Eloss'].to_numpy()
                w = scifi['Time'].to_numpy()
                bin_x = np.linspace(x_range[0], x_range[1], 100 + 1)
                bin_y = np.linspace(y_range[0], y_range[1], 100 + 1)
                _hist(axes[0, 0], z, y, (bin_z, bin_y), w, 'y', 'SciFi: y–z')
                _hist(axes[0, 1], z, x, (bin_z, bin_x), w, 'x', 'SciFi: x–z')

            # Row 1: Scint
            if scint.empty:
                for c in (0, 1):
                    axes[1, c].text(0.5, 0.5, "No Scint hits", ha='center', va='center', transform=axes[1, c].transAxes)
                    axes[1, c].set_axis_off()
            else:
                z = scint['layer_id'].to_numpy()
                x = scint['x'].to_numpy()
                y = scint['y'].to_numpy()
                # w = scint['Eloss'].to_numpy()
                w = scint['Time'].to_numpy()
                bin_x = np.linspace(x_range[0], x_range[1], nbins + 1)
                bin_y = np.linspace(y_range[0], y_range[1], nbins + 1)
                _hist(axes[1, 0], z, y, (bin_z, bin_y), w, 'y', 'Scint: y–z')
                _hist(axes[1, 1], z, x, (bin_z, bin_x), w, 'x', 'Scint: x–z')

            axes[1, 0].set_xlabel('z (layer_id)')
            axes[1, 1].set_xlabel('z (layer_id)')
            fig.suptitle(f'Event {event} True Hits (2×2)', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = outfile2x2.format(event=event)
            fig.savefig(path)
            print(f"Saved: {path}")

        elif mode == "3d":
            fig = plt.figure(figsize=(18, 9))
            # SciFi 3D
            ax0 = fig.add_subplot(1, 2, 1, projection='3d')
            if scifi.empty:
                ax0.text2D(0.5, 0.5, "No SciFi hits", transform=ax0.transAxes, ha='center', va='center')
                ax0.set_axis_off()
            else:
                x, y, z, w = (scifi['x'], scifi['y'], scifi['layer_id'], scifi['Eloss'])
                sz = 50 * (w / w.max()) if w.max() > 0 else np.full_like(w, 10)
                sc = ax0.scatter(x, y, z, c=w, s=sz, cmap='viridis', norm=norm, alpha=0.8)
                ax0.set_title('SciFi True Hits')
                ax0.set_xlabel('x'); ax0.set_ylabel('y'); ax0.set_zlabel('z')
                fig.colorbar(sc, ax=ax0, pad=0.1)

            # Scint 3D
            ax1 = fig.add_subplot(1, 2, 2, projection='3d')
            if scint.empty:
                ax1.text2D(0.5, 0.5, "No Scint hits", transform=ax1.transAxes, ha='center', va='center')
                ax1.set_axis_off()
            else:
                x, y, z, w = (scint['x'], scint['y'], scint['layer_id'], scint['Eloss'])
                sz = 50 * (w / w.max()) if w.max() > 0 else np.full_like(w, 10)
                sc = ax1.scatter(x, y, z, c=w, s=sz, cmap='viridis', norm=norm, alpha=0.8)
                ax1.set_title('Scint True Hits')
                ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
                fig.colorbar(sc, ax=ax1, pad=0.1)

            fig.suptitle(f'Event {event} True Hits (3D)', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            path = outfile3d.format(event=event)
            fig.savefig(path)
            print(f"Saved: {path}")

        else:
            raise ValueError("mode must be '2x2' or '3d'")

    def plot_both(self, event=0, merged=True):
        # select event
        df_evt = self.df_true[self.df_true['event_id'] == event]
        if df_evt.empty:
            print(f"No entries for event {event} in df_true")
            return

        # split SciFi vs Scint
        scifi = df_evt[df_evt['layer_type'].isin([0, 1])]
        scint = df_evt[df_evt['layer_type'] == 2]
        print("TRUE EVENT: ", scifi.columns)
        df_evt_digi = self.df_merged[self.df_merged['event_id'] == event]
        if df_evt_digi.empty:
            print(f"No entries for event {event} in df")
            return
        # split SciFi vs Scint
        # print("DIGI EVENT: ", df_evt_digi)
        scifi_digi = df_evt_digi[df_evt_digi['layer_type'].isin([0, 1])]
        scint_digi = df_evt_digi[df_evt_digi['layer_type'] == 2]
        print("DIGI EVENT: ", scifi_digi.columns)
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))
        bins = np.linspace(-25, 25, 100)
        ax[0].hist(scifi["y"], weights = scifi["Eloss"], bins=bins, histtype='step', density= True, label=f"True y: {len(scifi['y'])} entries", color='blue')
        ax[0].hist(scifi_digi["y_digi"], weights = scifi_digi["Signal"], bins=bins, histtype='step', density= True, label=f"Digi y: {len(scifi_digi['y_digi'])} entries", color='orange')
        ax[0].set_xlabel('y [cm]')
        ax[0].set_ylabel('Counts')
        ax[0].set_title(f'Scifi')
        ax[0].legend()
        bins = np.linspace(-25, 25, 50)
        ax[1].hist(scint["y"], bins=bins, histtype='step', label='True y', color='blue')
        ax[1].hist(scint_digi["y_digi"], bins=bins, histtype='step', label='Digi y', color='orange')
        ax[1].set_xlabel('y [cm]')
        ax[1].set_ylabel('Counts')
        ax[1].set_title(f'Scint')
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(f"figures/true_vs_digi_y_event_{event}.pdf")


    def vis_digi_multiplicity(self):
        """
        Visualize the multiplicity of hits in each event.
        This function creates a histogram of the number of hits per event.
        """
        import matplotlib.pyplot as plt

        # Count the number of hits per event
        hit_counts = self.df.groupby('event_id').size().reset_index()
        hit_signals = self.df.groupby('event_id').sum().reset_index()['Signal']
        nu_e = self.df.groupby('event_id').mean().reset_index()
        print(nu_e)
        df_1 = pd.DataFrame({"event_id": nu_e["event_id"], "hit_counts": hit_counts[0], "hit_signals": hit_signals, "nu_e": nu_e['E_nu']})
        bins = 50
        df_1 = df_1.query(f"hit_counts < {bins} & nu_e > 1.")
        print(df_1["event_id"])
        # Create a histogram of hit counts
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].hist(df_1["hit_counts"], bins=bins, color='blue', alpha=0.7)
        ax[0].set_xlabel('Number of Hits per Event')
        ax[0].set_ylabel('Number of Events')
        ax[0].set_title('Multiplicity of Hits in Each Event')
        h = ax[1].hist2d(df_1["hit_counts"], df_1["nu_e"], bins=bins, cmap='viridis', norm=LogNorm())
        ax[1].set_xlabel('Number of Hits per Event')
        ax[1].set_ylabel('Neutrino Energy [GeV]')
        fig.colorbar(h[3], ax=ax[1])
        ax[1].set_title('Multiplicity vs Neutrino Energy')
        h = ax[2].hist2d(df_1["hit_signals"], df_1["nu_e"], bins=bins, cmap='viridis', norm=LogNorm())
        ax[2].set_xlabel('Signal per Event')
        ax[2].set_ylabel('Neutrino Energy [GeV]')
        fig.colorbar(h[3], ax=ax[2])
        ax[2].set_title('Multiplicity vs Neutrino Energy')
        fig.tight_layout()
        fig.savefig('multiplicity_vs_energy.pdf')


    def store_events_3_histograms_digi(
        self, filename="event_3_histograms.root", plot_type="hist2d"
    ):
        """
        Process the digitized DataFrame and, for each event, build three 2D numpy histograms:
        - Histogram for layer_type == 1 using digitized layer_id and fiber_id_local with fixed binning.
        - Histogram for layer_type == 2 using digitized layer_id and fiber_id_local with fixed binning.
        - Histogram for layer_type == 3 (scintillator) using layer_id and fiber_id_local with specified bins.

        For layer_type 1 and 2, if plot_type is "hist2d", the hit weights are given by the energy loss (Eloss),
        just as is done for layer_type 3. In "scatter" mode the histograms are produced without weights.

        The SciFi-style histograms (for types 1 and 2) are built with:
        - X-axis: digitized layer_id using fixed bins (here we assume 45 layers).
        - Y-axis: digitized fiber_id_local with a number of bins set by self.number_of_fibers.

        The scintillator histogram (for type 3) is built with:
        - X-axis: layer_id (assuming 45 layers).
        - Y-axis: fiber_id_local with 2500 units range (2501 bin edges).

        The histograms (flattened to 1D arrays) and event-related quantities (E_nu, theta_nu, and pdg_nu)
        are stored into a ROOT file (using uproot) as a TTree with branches:
        - event_id, hist_type1, hist_type2, hist_type3, E_nu, theta_nu, and nu_flavor.

        Parameters:
        filename  (str): The ROOT file name to store the histograms.
        plot_type (str): Either "hist2d" (with Eloss weights) or "scatter" (unweighted).
        """


        # --- Define fixed binning for layer_type 1 and 2 (SciFi-style) ---
        bins_scifi_x = np.arange(-0.5, self.detector_properties["nLayers"] + 0.5, 1.0)
        bins_scifi_y = np.arange(-0.5, 128*7 + 0.5, 1.0)
        scifi_shape = (len(bins_scifi_x) - 1, len(bins_scifi_y) - 1)

        # --- Define fixed binning for layer_type 3 (Scintillator-style) ---
        bins_scint_x = np.arange(-0.5, self.detector_properties["nLayers"] + 0.5, 1.0)
        bins_scint_y = np.linspace(0, 2500, 2501)
        scint_shape = (len(bins_scint_x) - 1, len(bins_scint_y) - 1)

        # --- Prepare lists to hold histogram data for each event ---
        event_ids = []
        hist_type1 = []
        hist_type2 = []
        hist_type3 = []
        E_nu = []
        theta_nu = []
        nu_flavor = []

        # Group the DataFrame by event_id so we process one event at a time.
        grouped = self.df.groupby("event_id")

        for event_id, df_event in grouped:
            if len(df_event["E_nu"]) < 10:
                print(f"Skipping event {event_id} with insufficient data.")
                continue
            event_ids.append(event_id)
            E_nu.append(df_event["E_nu"].iloc[0])
            theta_nu.append(df_event["theta_nu"].iloc[0])
            nu_flavor.append(df_event["pdg_nu"].iloc[0])

            # --- Build histogram for layer_type == 1 ---
            df_type1 = df_event[df_event["layer_type"] == 0]
            if not df_type1.empty:
                if plot_type == "hist2d":
                    H_type1, _, _ = np.histogram2d(
                        df_type1["layer_id"],
                        ((df_type1['local_channel_id'] % 10000) // 1000) * 128 + (df_type1['local_channel_id'] % 1000),
                        bins=[bins_scifi_x, bins_scifi_y],
                        weights=df_type1["Signal"],
                    )
                elif plot_type == "scatter":
                    H_type1, _, _ = np.histogram2d(
                        df_type1["layer_id"],
                        df_type1["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type1 = np.zeros(scifi_shape)
            hist_type1.append(H_type1.flatten())

            # --- Build histogram for layer_type == 2 ---
            df_type2 = df_event[df_event["layer_type"] == 1]
            if not df_type2.empty:
                if plot_type == "hist2d":
                    H_type2, _, _ = np.histogram2d(
                        df_type2["layer_id"],
                        ((df_type2['local_channel_id'] % 10000) // 1000) * 128 + (df_type2['local_channel_id'] % 1000),
                        bins=[bins_scifi_x, bins_scifi_y],
                        weights=df_type2["Signal"],
                    )
                elif plot_type == "scatter":
                    H_type2, _, _ = np.histogram2d(
                        df_type2["layer_id"],
                        df_type2["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type2 = np.zeros(scifi_shape)
            hist_type2.append(H_type2.flatten())

            # --- Build histogram for layer_type == 3 ---
            df_type3 = df_event[df_event["layer_type"] == 2]
            if not df_type3.empty:
                if plot_type == "hist2d":
                    H_type3, _, _ = np.histogram2d(
                        df_type3["layer_id"],
                        df_type3['local_channel_id'],
                        bins=[bins_scint_x, bins_scint_y],
                        weights=df_type3["Signal"],
                    )
                elif plot_type == "scatter":
                    H_type3, _, _ = np.histogram2d(
                        df_type3["layer_id"],
                        df_type3["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y],
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type3 = np.zeros(scint_shape)
            hist_type3.append(H_type3.flatten())

        # Convert lists to numpy arrays for storage.
        event_ids = np.array(event_ids, dtype=np.int32)
        hist_type1 = np.array(hist_type1, dtype=np.float32)
        hist_type2 = np.array(hist_type2, dtype=np.float32)
        hist_type3 = np.array(hist_type3, dtype=np.float32)

        # --- Write the histograms and additional variables to a ROOT file ---
        # Create a dictionary to be written as a TTree.
        tree_data = {
            "event_id": event_ids,
            "scifimat_1": hist_type1,
            "scifimat_2": hist_type2,
            "scint": hist_type3,
            "E_nu": E_nu,
            "theta_nu": theta_nu,
            "nu_flavor": nu_flavor,
        }

        # Use uproot to create (or overwrite) the ROOT file with a TTree named "EventTree".
        with uproot.recreate(filename) as root_file:
            root_file["EventTree"] = tree_data




def configure_snd_mtc(yaml_file):
    with open(yaml_file) as file:
        config = yaml.safe_load(file)
        return config

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Analyze FairShip data")
    parser.add_argument("--event_id", type=int, default=22, help="Event ID to display")
    parser.add_argument(
        "--plot_type",
        type=str,
        default="hist2d",
        choices=["hist2d", "scatter"],
        help="Type of plot",
    )
    parser.add_argument(
        "--plot_coords",
        type=str,
        default="both",
        choices=["digi", "zx", "both", "3d"],
        help="Coordinate system(s) to plot",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dummy",
        choices=["display", "store", "read", "dummy"],
        help="Mode of operation",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=str,
        default="ship.conical.Genie-TGeant4.root",
        help="Input filename (used in single mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="event_histograms.root",
        help="Output filename (used in single mode)",
    )
    parser.add_argument(
        "--mode_batch",
        type=str,
        default="single",
        choices=["single", "batch"],
        help="Run on one file or batch of 1–1000",
    )
    parser.add_argument(
        "--batch_flavor",
        type=str,
        default="14",
        choices=["12", "14", "16"],
        help="Choose flavor",
    )
    parser.add_argument("--geofile", type=str, default="geofile_full.conical.PG_211-TGeant4.root", help="Geofile to use")

    args = parser.parse_args()


    fairship = os.environ["FAIRSHIP"]

    params = configure_snd_mtc(
            os.path.join(fairship, "geometry", "MTC_config.yaml")
        )["MTC"]

    logging.info(f"Using parameters: {params}")
    if args.mode_batch == "batch":
        # Use brace expansion for 1-100 (works in bash, but glob.glob does not expand braces)
        # So we generate the list in Python:
        pattern_list = [
            f"/eos/experiment/ship/user/edursov/pycondor_out/nuCCDIS_june/{args.batch_flavor}/{i}/ship.conical.Genie-TGeant4_rec.root"
            for i in range(1, 1001)
        ]
        input_files = [f for f in pattern_list if os.path.isfile(f)]
        # input_files = sorted(glob.glob(pattern))
        # if not input_files:
        #     print(f"No files found matching pattern: {pattern}")
        for input_path in input_files:
            dir_path = os.path.dirname(input_path)
            output_path = os.path.join(dir_path, "event_3_histograms_10_hist_40x40.root")
            if not os.path.isfile(input_path):
                print(f"Skipping: {input_path} not found.")
                continue
            print(f"Processing {input_path}")
            run_single(input_path, output_path, args, params)
    else:
        print("check")
        run_single(args.input, args.output, args, params)



def run_single(input_file, output_file, args, params):

 

    analyzer = FairShipAnalyzer(input_file, params)
    if analyzer.chain is None:
        print(f"Error: Unable to open file {input_file}.")
        return
    # analyzer.create_fiber_structure(hardcoded=args.hardcoded)
    analyzer.read_true()
    analyzer.read_geo(args.geofile)
    analyzer.read_digi()
    print("check")
    if args.mode == "display":
        print("check")
        analyzer.study_clustering()
        analyzer.study_clustering_3d()
        analyzer.study_clustering_2d()
        analyzer.plot_clustering_efficiency(args.event_id)
        # analyzer.event_display_digi_new(args.event_id, args.plot_type, args.plot_coords)
        analyzer.vis_true(event=args.event_id, mode="2x2")
        # analyzer.vis_true(event=args.event_id, mode="3d")
        # # analyzer.event_display(args.event_id, coord="z")
        # # analyzer.event_display_1(args.event_id, coord="3d")
        # # for idx, event in events:
        # # analyzer.vis_digi_1(event=args.event_id)
        # # analyzer.vis_digi_multiplicity()
        analyzer.reconstruct_scifi_digihit_position(clustering=True)
        analyzer.merge_back_scint()
        # # After merging pairs + scint:
        analyzer.vis_digi_1(event=args.event_id, merged = True, make_3d=False, two_by_two=True)
        analyzer.plot_both(event=args.event_id, merged=True)
        # Original 3×2 view:
        # analyzer.vis_digi_1(event=args.event_id)

        # 3D view:
        # analyzer.vis_digi_1(event=args.event_id, make_3d=True)
    elif args.mode == "store":
        # analyzer.store_events_3_histograms(
        #     filename=output_file, plot_type=args.plot_type
        # )
        analyzer.store_events_3_histograms_digi(
            filename=output_file, plot_type=args.plot_type
        )
        # analyzer.store_df_digi(filename=f"df_digi_{args.batch_flavor}.pkl")
    elif args.mode == "read":
        analyzer.read_events_histograms(filename=output_file)
    else:
        analyzer.analyze_events()


if __name__ == "__main__":
    main()
