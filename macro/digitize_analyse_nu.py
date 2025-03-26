import ROOT as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Import LogNorm for log scale colorbars
import seaborn as sns
sns.set_style("whitegrid")



class FairShipAnalyzer:
    def __init__(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        self.file_list = file_list
        self.chain = self._create_chain()

    def _create_chain(self):
        ch = r.TChain('cbmsim')
        for file in self.file_list:
            ch.Add(str(file))
        return ch

    def get_entries(self):
        return self.chain.GetEntries()

    def get_branches(self):
        return self.chain.GetListOfBranches()

    def analyze_events(self):
        for i, event in enumerate(self.chain):
            print(i, len(event.MTCdetPoint))


    def set_fiber_dimensions(self, detector_dimensions, fiber_dimensions):
        self.fiber_dimensions = fiber_dimensions
        self.detector_dimensions = detector_dimensions
        self.detector_dimensions["angle"] = np.radians(self.detector_dimensions["angle"])
        self.detector_dimensions["width"] = self.detector_dimensions["width"] - self.detector_dimensions["width"] * np.tan(self.detector_dimensions["angle"])
        self.fiber_dimensions["fiber_length"] = self.detector_dimensions["height"] * np.cos(self.detector_dimensions["angle"])

    def get_local_fiber_id(self, hit):
        # Retrieve the detector tilt angle (in radians)
        angle = self.detector_dimensions["angle"] if hit.GetLayerType() == 1 else -self.detector_dimensions["angle"]
        # Project the hit position (x, y) onto the fiber pitch direction.
        # This gives the effective coordinate across the fibers.
        local_u = hit.GetX() * np.cos(angle) + hit.GetY() * np.sin(angle)
        
        # Create segments along the projected width using fiber pitch
        plane_segments = np.arange(-self.detector_dimensions["width"]/(2*np.cos(angle)), self.detector_dimensions["width"]/(2*np.cos(angle)), self.fiber_dimensions["fiber_pitch"])
        self.number_of_fibers = len(plane_segments)
        # Identify the fiber index corresponding to the projected hit position
        fiber_id = np.digitize(local_u, plane_segments) - 1
        return fiber_id
    
    def get_global_fiber_id(self, hit):
        local_fiber_id = self.get_local_fiber_id(hit)
        global_fiber_id = hit.GetDetectorID() + local_fiber_id
        return global_fiber_id

    def reconstruct_hit_position(self, det_id_1, det_id_2):
        """
        Reconstruct the (x,y) coordinate where a particle crossed the two fiber planes.
        
        Parameters:
        det_id_1: Fiber ID from the first plane (tilted by +5 degrees)
        det_id_2: Fiber ID from the second plane (tilted by -5 degrees)
        
        Returns:
        (x, y): The reconstructed coordinate of the particle hit.
        """
        # Define the tilt angles in radians
        angle1 = np.radians(5)   # first plane tilt +5°
        angle2 = np.radians(-5)  # second plane tilt -5°
        
        # Retrieve fiber pitch (assumed to be defined in self.fiber_dimensions)
        pitch = self.fiber_dimensions["fiber_pitch"]
    
        # For each plane, the actual hit position is unknown within the fiber.
        # We assume a uniform distribution within the fiber width.
        u1 = det_id_1 * pitch + np.random.uniform(0, pitch)
        u2 = det_id_2 * pitch + np.random.uniform(0, pitch)
        # For each plane, the fiber line is given by:
        # Plane 1: x*cos(angle1) + y*sin(angle1) = u1
        # Plane 2: x*cos(angle2) + y*sin(angle2) = u2
        # We form the linear system A * [x, y]^T = b and solve for x and y.
        A = np.array([[np.cos(angle1), np.sin(angle1)],
                    [np.cos(angle2), np.sin(angle2)]])
        b = np.array([u1, u2])
        
        # Solve for (x, y)
        x, y = np.linalg.solve(A, b)
        if np.abs(x) < self.detector_dimensions["width"] / 2.0 and np.abs(y) < self.detector_dimensions["height"] / 2.0:
            return (x, y)
        else:
            return None

    def event_display(self, event_id, coord="z"):
        """
        Display an event by plotting two 2D histograms showing the projections of the hit positions.
        The user can choose to plot against the Z coordinate (default) or the layer number.

        Parameters:
        event_id: The identifier for the event.
        coord: Either "z" (to use the Z coordinate) or "layer" (to use the layer number).
        """
        df_event = self.df_digi.query(f"event_id == {event_id}")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        if coord == "z":
            x_data = df_event["z"]
            x_label = "Z [cm]"
            # Create 100 bins between the start and end of the detector in Z.
            bins_x = np.linspace(self.detector_dimensions["startZ"],
                                self.detector_dimensions["endZ"], 100)
        elif coord == "layer":
            # Here we assume that the dataframe has a 'layer' column.
            x_data = df_event["layer_id"]
            x_label = "Layer N"
            # Use bins that correctly bin discrete layers (assuming layers are integer-valued)
            bins_x = np.arange(0 - 0.5, 44 + 1.5, 1)
        else:
            raise ValueError("coord must be either 'z' or 'layer'")

        # Define bins for the second coordinate in each plot
        bins_1 = (bins_x, np.linspace(-self.detector_dimensions["width"]/2,
                                        self.detector_dimensions["width"]/2, 100))
        bins_2 = (bins_x, np.linspace(-self.detector_dimensions["height"]/2,
                                        self.detector_dimensions["height"]/2, 100))

        # Plot Z(or layer)-X projection.
        ax[0].hist2d(x_data, df_event["x"], bins=bins_1, norm=LogNorm(), cmap="viridis")
        ax[0].set_title(f"{x_label}-X projection")
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel("X [cm]")

        # Plot Z(or layer)-Y projection.
        ax[1].hist2d(x_data, df_event["y"], bins=bins_2, norm=LogNorm(), cmap="viridis")
        ax[1].set_title(f"{x_label}-Y projection")
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel("Y [cm]")

        # Debug print to show the range in Z from the complete dataset
        print(self.df_digi["z"].min(), self.df_digi["z"].max())
        fig.savefig(f"event_display_{coord}.png")

    def event_display_digi(self, event_id):
        """
        Display an event using two subplots:
        
        Plot 1 (Left): For layer_type 1 or 2, a 2D histogram of fiber_id_local (Y-axis) vs. layer_id (X-axis).
                    The fiber_id_local is expected to range from 0 to self.number_of_fibers*2.
        
        Plot 2 (Right): For layer_type 3, a 2D histogram of fiber_id_local (Y-axis) vs. layer_id (X-axis),
                    with fiber_id_local ranging from 0 to 50*50, and the histogram is weighted by the energy loss (Eloss).
        
        Assumptions:
        - self.df_digi is the dataframe created by create_fiber_structure and now includes an "Eloss" column.
        - self.number_of_fibers is defined in the class.
        """
        # Filter the dataframe to select the event of interest
        df_event = self.df_digi[self.df_digi["event_id"] == event_id]
        
        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi = 100)
        
        # -------------------------
        # Plot 1: For layer_type 1 or 2
        # -------------------------
        df_12 = df_event[df_event["layer_type"].isin([1, 2])]
        
        # For the X-axis, we use layer_id which is assumed to be integer-valued.
        # Create bins for layer_id with a half-bin offset for proper binning.
        # if not df_12.empty:
        #     layer_min = int(df_12["layer_id"].min())
        #     layer_max = int(df_12["layer_id"].max())
        # else:
        #     layer_min, layer_max = 0, 1
        bins_x1 = np.arange(0 - 0.5, 45*2 + 0.5, 1.0)
        
        # For the Y-axis, fiber_id_local ranges from 0 to self.number_of_fibers * 2.
        # bins_y1 = np.linspace(0, self.number_of_fibers * 2, self.number_of_fibers * 2 + 1)
        bins_y1 = np.linspace(0, self.number_of_fibers, self.number_of_fibers + 1)
        
        h1 = ax[0].hist2d(df_12["layer_id"], df_12["fiber_id_local"], bins=[bins_x1, bins_y1], norm=LogNorm(),
                        cmap=plt.cm.jet)
        ax[0].set_xlabel("Layer ID")
        ax[0].set_ylabel("Fiber ID")
        ax[0].set_title("Scifi. " + f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]")
        # ax[0].legend()
        # fig.colorbar(h1[3], ax=ax[0])
        
        # -------------------------
        # Plot 2: For layer_type 3
        # -------------------------
        df_3 = df_event[df_event["layer_type"] == 3]
        
        # if not df_3.empty:
        #     layer_min_3 = int(df_3["layer_id"].min())
        #     layer_max_3 = int(df_3["layer_id"].max())
        # else:
        #     layer_min_3, layer_max_3 = 0, 1
        bins_x2 = np.arange(0 - 0.5, 44 + 1.5, 1.0)
        
        # For layer_type 3, fiber_id_local is expected to go from 0 to 50*50 (2500).
        # Here we use 50 bins for fiber_id_local.
        bins_y2 = np.linspace(0, 2500, 100 + 1)
        
        # Weight the histogram by the "Eloss" column.
        h2 = ax[1].hist2d(df_3["layer_id"], df_3["fiber_id_local"], bins=[bins_x2, bins_y2], norm=LogNorm(),
                        weights=df_3["Eloss"], cmap=plt.cm.jet)
        ax[1].set_xlabel("Layer ID")
        ax[1].set_ylabel("Cell ID")
        ax[1].set_title("Scint. " + f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]")



    def event_display_digi_new(self, event_id, plot_type="hist2d", plot_coords="both"):
        """
        Display an event using subplots in one or two coordinate systems.

        Two coordinate systems are supported:
        - "digi": The digitized coordinate system.
            • SciFi (layer_type 1 or 2): 2D histogram (or scatter) of fiber_id_local (Y) vs. layer_id (X).
            • Scintillator (layer_type 3): 2D histogram (or scatter) of fiber_id_local (Y) vs. layer_id (X),
                with the histogram weighted by energy loss.
            • An arrow (with label) is drawn on the SciFi plot showing the neutrino’s incoming direction
                (using mapping functions to go from real coordinates to digitized ones).

        - "zx": The physical (real) coordinate system (Z vs. X).
            • For SciFi, the digitized fiber_id_local is converted back into a real x coordinate.
                The layer_id is inverted to a real z coordinate assuming a linear mapping.
            • For Scintillator, an assumed mapping converts the cell ID (fiber_id_local) into a real x value,
                and the layer_id into z.
            • The neutrino arrow is drawn at (x_nu, z_nu), with its tail computed from the momentum.

        - "both": Both coordinate systems are displayed in one canvas arranged in two rows.
            The top row is the digitized view and the bottom row is the Z–X view.

        Parameters:
        event_id (int): ID of the event to display.
        plot_type (str): Either "hist2d" or "scatter" for the type of plot.
        plot_coords (str): Which coordinate system(s) to plot: "digi", "zx", or "both".
        """


        # --- Helper mapping functions for digitized coordinates (already provided) ---
        def map_z_to_layer(z, startZ, endZ, total_layers):
            """Map a real z coordinate to a digitized layer id."""
            return (z - startZ) / (endZ - startZ) * total_layers

        def map_x_to_fiber(x):
            """
            Map a real x coordinate to a fiber id.
            Uses the same procedure as get_local_fiber_id.
            """
            angle = self.detector_dimensions["angle"]
            width = self.detector_dimensions["width"]
            fiber_pitch = self.fiber_dimensions["fiber_pitch"]
            segments = np.arange(-width/(2*np.cos(angle)),
                                width/(2*np.cos(angle)),
                                fiber_pitch)
            # np.digitize returns an index; subtract one to get a zero-based fiber id.
            fiber_id = np.digitize([x], segments) - 1
            return fiber_id[0]

        def particle_map(pdg):
            """Map a PDG code to a LaTeX formatted particle name for annotation."""
            if pdg == 12:
                return "${\\nu}_{e}$"
            elif pdg == 14:
                return "${\\nu}_{\\mu}$"
            elif pdg == 16:
                return "${\\nu}_{\\tau}$"
            elif pdg == -12:
                return "${\\bar{\\nu}}_{e}$"
            elif pdg == -14:
                return "${\\bar{\\nu}}_{\\mu}$"
            elif pdg == -16:
                return "${\\bar{\\nu}}_{\\tau}$"
            else:
                return f"PDG {pdg}"



        # --- Retrieve the event data ---
        df_event = self.df_digi[self.df_digi["event_id"] == event_id]
        # For annotation, get neutrino parameters once.
        x_nu = df_event["x_nu"].iloc[0]
        z_nu = df_event["z_nu"].iloc[0]
        px_nu = df_event["px_nu"].iloc[0]
        pz_nu = df_event["pz_nu"].iloc[0]
        particle_name = particle_map(df_event["pdg_nu"].iloc[0])

        # Common detector parameters
        total_sciFi_layers = 45 * 2  # 90 layers for SciFi digitization
        startZ = self.detector_dimensions["startZ"]
        endZ = self.detector_dimensions["endZ"]
        width = self.detector_dimensions["width"]
        angle = self.detector_dimensions["angle"]
        fiber_pitch = self.fiber_dimensions["fiber_pitch"]

        # Determine figure layout based on chosen coordinate system(s)
        if plot_coords == "both":
            fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=100)
            ax_digi_scifi = ax[0, 0]
            ax_digi_scint = ax[0, 1]
            ax_zx_scifi = ax[1, 0]
            ax_zx_scint = ax[1, 1]
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
            if plot_coords == "digi":
                ax_digi_scifi = ax[0]
                ax_digi_scint = ax[1]
            elif plot_coords == "zx":
                ax_zx_scifi = ax[0]
                ax_zx_scint = ax[1]
            else:
                raise ValueError("plot_coords must be 'digi', 'zx', or 'both'")

        # -------------------------
        # Plot the Digitized Coordinates if requested
        # -------------------------
        if plot_coords in ["digi", "both"]:
            # --- SciFi (layer_type 1 or 2) digitized ---
            df_12 = df_event[df_event["layer_type"].isin([1, 2])]
            bins_x1 = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
            bins_y1 = np.linspace(0, self.number_of_fibers, self.number_of_fibers + 1)
            if plot_type == "hist2d":
                ax_digi_scifi.hist2d(df_12["layer_id"], df_12["fiber_id_local"],
                                    bins=[bins_x1, bins_y1],
                                    norm=LogNorm(), cmap=plt.cm.jet)
            elif plot_type == "scatter":
                ax_digi_scifi.scatter(df_12["layer_id"], df_12["fiber_id_local"],
                                    color="blue", marker="o", label="SciFi", alpha=0.7)
            else:
                raise ValueError("plot_type must be either 'hist2d' or 'scatter'")
            ax_digi_scifi.set_xlabel("Layer ID")
            ax_digi_scifi.set_ylabel("Fiber ID")
            ax_digi_scifi.set_title("SciFi (Digitized). " +
                                    f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]. Event multiplicity: {df_event['multiplicity'].iloc[0]}")

            # --- Draw the neutrino arrow on the SciFi digitized plot ---
            tip_layer = map_z_to_layer(z_nu, startZ, endZ, total_sciFi_layers)
            tip_fiber = map_x_to_fiber(x_nu) + 0.5  # center of fiber bin
            # Conversion factors for digitized coordinates.
            d_layer_per_dz = total_sciFi_layers / (endZ - startZ)
            d_fiber_per_dx = 1 / fiber_pitch
            scale = 1  # adjust as needed for visual clarity
            delta_layer = scale * pz_nu * d_layer_per_dz
            delta_fiber = scale * px_nu * d_fiber_per_dx
            tail_layer = tip_layer - delta_layer
            tail_fiber = tip_fiber - delta_fiber
            ax_digi_scifi.annotate("",
                                xy=(tip_layer, tip_fiber), xycoords='data',
                                xytext=(tail_layer, tail_fiber), textcoords='data',
                                arrowprops=dict(arrowstyle="->", color="red", lw=2))
            label_offset = 2
            ax_digi_scifi.text(tip_layer - 5, tip_fiber + label_offset, particle_name,
                            color="red", ha="center", va="bottom", fontsize=12)

            # --- Scintillator (layer_type 3) digitized ---
            df_3 = df_event[df_event["layer_type"] == 3]


            bins_x2 = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)  # assume 45 layers
            bins_y2 = np.linspace(0, 2500, 101)
            if plot_type == "hist2d":
                ax_digi_scint.hist2d(df_3["layer_id"], df_3["fiber_id_local"],
                                    bins=[bins_x2, bins_y2],
                                    norm=LogNorm(), weights=df_3["Eloss"], cmap=plt.cm.jet)
            elif plot_type == "scatter":
                ax_digi_scint.scatter(df_3["layer_id"], df_3["fiber_id_local"],
                                    c=df_3["Eloss"], norm=LogNorm(), cmap=plt.cm.jet,
                                    marker="o", label="Scint")
            ax_digi_scint.set_xlabel("Layer ID")
            ax_digi_scint.set_ylabel("Cell ID")
            ax_digi_scint.set_title("Scintillator (Digitized). " +
                                    f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]")

        # -------------------------
        # Plot the Real (Z-X) Coordinates if requested
        # -------------------------
        if plot_coords in ["zx", "both"]:
            # --- SciFi in real coordinates ---
            # For SciFi hits, convert digitized fiber_id_local back to real x and layer_id to real z.
            df_12 = df_event[df_event["layer_type"].isin([1, 2])].copy()
            df_12["x_real"] = df_12["x"]
            df_12["z_real"] = df_12["z"]
            bins_x1 = np.linspace(self.detector_dimensions["startZ"], self.detector_dimensions["endZ"], 100)
            bins_y1 = np.linspace(-self.detector_dimensions["width"]/2, self.detector_dimensions["width"]/2, 100)
            if plot_type == "hist2d":
                # Using 50 bins (adjust as needed)
                ax_zx_scifi.hist2d(df_12["z_real"], df_12["x_real"],
                                bins=(bins_x1, bins_y1), norm=LogNorm(), cmap=plt.cm.jet)
            elif plot_type == "scatter":
                ax_zx_scifi.scatter(df_12["z_real"], df_12["x_real"],
                                    color="blue", marker="o", label="SciFi", alpha=0.7)
            else:
                raise ValueError("plot_type must be either 'hist2d' or 'scatter'")
            ax_zx_scifi.set_xlabel("Z [cm]")
            ax_zx_scifi.set_ylabel("X [cm]")
            ax_zx_scifi.set_title("SciFi (Real). " +
                                f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]")

            # Draw the neutrino arrow in real coordinates.
            # Here we use the actual neutrino coordinates; for a simple arrow the tail is computed directly.
            scale_real = 1  # adjust scale if needed
            delta_z = scale_real * pz_nu
            delta_x = scale_real * px_nu
            tail_z = z_nu - delta_z
            tail_x = x_nu - delta_x
            ax_zx_scifi.annotate("",
                                xy=(z_nu, x_nu), xycoords='data',
                                xytext=(tail_z, tail_x), textcoords='data',
                                arrowprops=dict(arrowstyle="->", color="red", lw=2))
            ax_zx_scifi.text(z_nu - 5, x_nu + label_offset, particle_name,
                            color="red", ha="center", va="bottom", fontsize=12)
            ax_zx_scint.annotate("",
                                xy=(z_nu, x_nu), xycoords='data',
                                xytext=(tail_z, tail_x), textcoords='data',
                                arrowprops=dict(arrowstyle="->", color="red", lw=2))
            ax_zx_scint.text(z_nu - 5, x_nu + label_offset, particle_name,
                            color="red", ha="center", va="bottom", fontsize=12)

            # --- Scintillator in real coordinates ---
            df_3 = df_event[df_event["layer_type"] == 3].copy()
            # For scintillator, map cell id (fiber_id_local) to real x.
            df_3["x_real"] = df_3["x"]
            # Map the layer_id (assumed to run from 0 to 44) to real z.
            df_3["z_real"] = df_3["z"]
            if plot_type == "hist2d":
                ax_zx_scint.hist2d(df_3["z_real"], df_3["x_real"],
                                bins=(bins_x1, bins_y1), norm=LogNorm(), weights=df_3["Eloss"], cmap=plt.cm.jet)
            elif plot_type == "scatter":
                ax_zx_scint.scatter(df_3["z_real"], df_3["x_real"],
                                    c=df_3["Eloss"], norm=LogNorm(), cmap=plt.cm.jet,
                                    marker="o", label="Scint")
            ax_zx_scint.set_xlabel("Z [cm]")
            ax_zx_scint.set_ylabel("X [cm]")
            ax_zx_scint.set_title("Scintillator (Real). " +
                                f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]")

        plt.tight_layout()
        fig.savefig("event_display_digi_new.pdf")

    def create_fiber_structure(self):
        df_digi = {"event_id": [], "multiplicity": [], "track_id": [], "E_nu": [], "px_nu": [], "pz_nu": [], "x_nu": [], "z_nu": [], "pdg_nu": [], "fiber_id": [], "fiber_id_local": [], "layer_type": [], "layer_id": [], "x": [], "y": [], "z": [], "pdg": [], "Eloss": []}
        for i, event in enumerate(self.chain):
            E_nu = np.sqrt(event.MCTrack[0].GetPx()**2 + event.MCTrack[0].GetPy()**2 + event.MCTrack[0].GetPz()**2)
            Px_nu, Pz_nu, X_nu, Z_nu = event.MCTrack[0].GetPx(), event.MCTrack[0].GetPz(), event.MCTrack[0].GetStartX(), event.MCTrack[0].GetStartZ()
            event_multiplicity = 0
            for mctrack in event.MCTrack:
                if mctrack.GetMotherId() == 0:
                    event_multiplicity += 1

            for hit in event.MTCdetPoint:
                if hit.GetLayerType() < 3:
                    fiber_id = self.get_global_fiber_id(hit)
                    fiber_id_local = self.get_local_fiber_id(hit)
                    if hit.GetEnergyLoss()*1e6 < 180:
                        continue
                    # fiber_id_local = fiber_id_local + self.number_of_fibers if hit.GetLayerType() == 2 else fiber_id_local
                else:
                    fiber_id = hit.GetDetectorID()
                    fiber_id_local = fiber_id % 10000

                df_digi["event_id"].append(i)
                df_digi["multiplicity"].append(event_multiplicity)
                df_digi["track_id"].append(hit.GetTrackID())
                df_digi["E_nu"].append(E_nu)
                df_digi["px_nu"].append(Px_nu)
                df_digi["pz_nu"].append(Pz_nu)
                df_digi["x_nu"].append(X_nu)
                df_digi["z_nu"].append(Z_nu)
                df_digi["pdg_nu"].append(event.MCTrack[0].GetPdgCode())
                df_digi["fiber_id"].append(fiber_id)
                df_digi["layer_type"].append(hit.GetLayerType())
                df_digi["fiber_id_local"].append(fiber_id_local)
                # df_digi["layer_id"].append(hit.GetLayer())
                if  hit.GetLayerType() == 3:
                    df_digi["layer_id"].append(hit.GetLayer() * 2 + 3)
                else:
                    df_digi["layer_id"].append(hit.GetLayer() * 2 if hit.GetLayerType() == 1 else hit.GetLayer() * 2 + 1)
                df_digi["x"].append(hit.GetX())
                df_digi["y"].append(hit.GetY())
                df_digi["z"].append(hit.GetZ())
                df_digi["pdg"].append(hit.PdgCode())
                df_digi["Eloss"].append(hit.GetEnergyLoss())
        df_digi = pd.DataFrame(df_digi)

        self.df_digi = df_digi
                



fileName = "ship.conical.Genie-TGeant4.root"
fiber_dimensions = {
    "fiber_pitch": 0.1,
    }
detector_dimensions = {
    "width": 50,
    "height": 50,
    "startZ": -3511.7150,
    "endZ":  -3175.5650,
    "angle": 5
}

analyzer = FairShipAnalyzer(fileName)
analyzer.set_fiber_dimensions(detector_dimensions, fiber_dimensions)

print(analyzer.get_entries())
print(analyzer.get_branches())
analyzer.create_fiber_structure()
analyzer.analyze_events()
# analyzer.event_display(22, coord = "layer")
analyzer.event_display_digi_new(22, "hist2d")




