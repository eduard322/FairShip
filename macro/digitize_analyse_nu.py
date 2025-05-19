import ROOT as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Import LogNorm for log scale colorbars
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import seaborn as sns
import argparse
import uproot
sns.set_style("whitegrid")



class FairShipAnalyzer:
    def __init__(self, file_list, detector_properties, fiber_dimensions):
        if isinstance(file_list, str):
            file_list = [file_list]
        self.file_list = file_list
        self.chain = self._create_chain()
        self.set_fiber_dimensions(detector_properties, fiber_dimensions)

    def _create_chain(self):
        ch = r.TChain('cbmsim')
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
            if len(event.MTCdetPoint) == 0:
                zero_events += 1
            print(i, len(event.MTCdetPoint))
        print(f"Number of events with no hits: {zero_events} out of {self.get_entries()}")

    def set_fiber_dimensions(self, detector_properties, fiber_dimensions):
        self.fiber_dimensions = fiber_dimensions
        self.detector_properties = detector_properties
        self.detector_properties["angle"] = np.radians(self.detector_properties["angle"])
        self.detector_properties["width"] = self.detector_properties["width"] - self.detector_properties["width"] * np.tan(self.detector_properties["angle"])
        self.fiber_dimensions["fiber_length"] = self.detector_properties["height"] * np.cos(self.detector_properties["angle"])

    def get_local_fiber_id(self, hit):
        # Retrieve the detector tilt angle (in radians)
        angle = self.detector_properties["angle"] if hit.GetLayerType() == 1 else -self.detector_properties["angle"]
        # Project the hit position (x, y) onto the fiber pitch direction.
        # This gives the effective coordinate across the fibers.
        local_u = hit.GetX() * np.cos(angle) + hit.GetY() * np.sin(angle)
        
        # Create segments along the projected width using fiber pitch
        plane_segments = np.arange(-self.detector_properties["width"]/(2*np.cos(angle)), self.detector_properties["width"]/(2*np.cos(angle)), self.fiber_dimensions["fiber_pitch"])
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
        if np.abs(x) < self.detector_properties["width"] / 2.0 and np.abs(y) < self.detector_properties["height"] / 2.0:
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
            bins_x = np.linspace(self.detector_properties["startZ"],
                                self.detector_properties["endZ"], 100)
        elif coord == "layer":
            # Here we assume that the dataframe has a 'layer' column.
            x_data = df_event["layer_id"]
            x_label = "Layer N"
            # Use bins that correctly bin discrete layers (assuming layers are integer-valued)
            bins_x = np.arange(0 - 0.5, 44 + 1.5, 1)
        else:
            raise ValueError("coord must be either 'z' or 'layer'")

        # Define bins for the second coordinate in each plot
        bins_1 = (bins_x, np.linspace(-self.detector_properties["width"]/2,
                                        self.detector_properties["width"]/2, 100))
        bins_2 = (bins_x, np.linspace(-self.detector_properties["height"]/2,
                                        self.detector_properties["height"]/2, 100))

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
        # print(self.df_digi["z"].min(), self.df_digi["z"].max())
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



    def event_display_digi_new(self, event_id, plot_type="hist2d", plot_coords="digi"):
        """
        Display an event using subplots in one or two coordinate systems, with an optional 3D view.

        Four display options are supported:
        - "digi": The digitized coordinate system.
            • SciFi (layer_type 1 or 2): 2D histogram (or scatter) of fiber_id_local (Y) vs. layer_id (X).
            • Scintillator (layer_type 3): 2D histogram (or scatter) of fiber_id_local (Y) vs. layer_id (X),
            with the histogram weighted by energy loss.
            • A neutrino arrow is drawn on the SciFi plot.
        - "zx": The physical (real) coordinate system (Z vs. X).
            • For SciFi, digitized fiber_id_local is converted back into a real x coordinate and layer_id
            is mapped to a real z.
            • For Scintillator, a similar mapping is assumed.
            • The neutrino arrow is drawn using the real neutrino coordinates.
        - "both": Both coordinate systems are shown in a 2×2 canvas.
        - "3d": A canvas with two rows is created.
            The top row is as in the "digi" display (2D digitized SciFi and Scintillator views).
            The second row is a 3D scatter plot of the event where:
                • x-axis: real Z (from df["z"])
                • y-axis: real X (from df["x"])
                • z-axis: an estimated fiber position computed as (fiber_id_local * fiber_pitch + fiber_pitch/2)
            The neutrino arrow is drawn using matplotlib’s 3D quiver (with no third-component of momentum).

        Parameters:
            event_id (int): ID of the event to display.
            plot_type (str): Either "hist2d" or "scatter" for the type of plot.
            plot_coords (str): Which coordinate system(s) to plot: "digi", "zx", "both", or "3d".
        """

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import numpy as np

        # --- Helper mapping functions for digitized coordinates (already provided) ---
        def map_z_to_layer(z, startZ, endZ, total_layers):
            """Map a real z coordinate to a digitized layer id."""
            return (z - startZ) / (endZ - startZ) * total_layers

        def map_x_to_fiber(x):
            """
            Map a real x coordinate to a fiber id.
            Uses the same procedure as get_local_fiber_id.
            """
            angle = self.detector_properties["angle"]
            width = self.detector_properties["width"]
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
        y_nu = df_event["y_nu"].iloc[0]
        z_nu = df_event["z_nu"].iloc[0]
        px_nu = df_event["px_nu"].iloc[0]
        py_nu = df_event["py_nu"].iloc[0]
        pz_nu = df_event["pz_nu"].iloc[0]
        particle_name = particle_map(df_event["pdg_nu"].iloc[0])

        # Common detector parameters
        total_sciFi_layers = 45 * 2  # 90 layers for SciFi digitization
        startZ = self.detector_properties["startZ"]
        endZ = self.detector_properties["endZ"]
        width = self.detector_properties["width"]
        angle = self.detector_properties["angle"]
        fiber_pitch = self.fiber_dimensions["fiber_pitch"]

        # Determine figure layout based on chosen coordinate system(s)
        if plot_coords == "both":
            fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=100)
            ax_digi_scifi = ax[0, 0]
            ax_digi_scint = ax[0, 1]
            ax_zx_scifi = ax[1, 0]
            ax_zx_scint = ax[1, 1]
        elif plot_coords == "3d":
            from matplotlib.gridspec import GridSpec
            fig = plt.figure(figsize=(12, 12), dpi=100)
            gs = GridSpec(2, 2, height_ratios=[1, 1])
            # Top row: digitized views (SciFi and Scintillator)
            ax_digi_scifi = fig.add_subplot(gs[0, 0])
            ax_digi_scint = fig.add_subplot(gs[0, 1])
            # Second row: 3D display spanning both columns.
            ax_3d = fig.add_subplot(gs[1, :], projection='3d')
        elif plot_coords in ["digi", "zx"]:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
            if plot_coords == "digi":
                ax_digi_scifi = ax[0]
                ax_digi_scint = ax[1]
            elif plot_coords == "zx":
                ax_zx_scifi = ax[0]
                ax_zx_scint = ax[1]
        else:
            raise ValueError("plot_coords must be 'digi', 'zx', 'both', or '3d'")

        # -------------------------
        # Plot the Digitized Coordinates if requested
        # -------------------------
        if plot_coords in ["digi", "both", "3d"]:
            # --- SciFi (layer_type 1 or 2) digitized ---
            df_12 = df_event[df_event["layer_type"].isin([1, 2])]
            bins_x1 = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
            bins_y1 = np.arange(0, self.number_of_fibers + 2)
            if plot_type == "hist2d":
                h = ax_digi_scifi.hist2d(df_12["layer_id"], df_12["fiber_id_local"],
                                        bins=[bins_x1, bins_y1],
                                        norm=LogNorm(), cmap=plt.cm.jet)
                print("scifi: ", h[0].shape)
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
            bins_x2 = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
            bins_y2 = np.arange(0, 2500 + 1)
            if plot_type == "hist2d":
                h = ax_digi_scint.hist2d(df_3["layer_id"], df_3["fiber_id_local"],
                                        bins=[bins_x2, bins_y2],
                                        norm=LogNorm(), 
                                        # weights=df_3["Eloss"], 
                                        cmap=plt.cm.jet)
                print("scint: ", h[0].shape)
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
            df_12 = df_event[df_event["layer_type"].isin([1, 2])].copy()
            df_12["x_real"] = df_12["x"]
            df_12["z_real"] = df_12["z"]
            bins_x1 = np.linspace(self.detector_properties["startZ"], self.detector_properties["endZ"], 100)
            bins_y1 = np.linspace(-self.detector_properties["width"]/2, self.detector_properties["width"]/2, 100)
            if plot_type == "hist2d":
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
            df_3["x_real"] = df_3["x"]
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

        # -------------------------
        # Plot the 3D Event Display if requested
        # -------------------------
        if plot_coords == "3d":
            # For 3D, we combine both SciFi and Scintillator hits.
            # Use real coordinates for X and Z, and estimate a fiber (Y) position from fiber_id_local.
            df_12 = df_event[df_event["layer_type"].isin([1,2])].copy()
            df_12["x_real"] = df_12["x"]
            df_12["z_real"] = df_12["z"]
            df_12["y_real"] = df_12["y"]

            df_3 = df_event[df_event["layer_type"] == 3].copy()
            df_3["x_real"] = df_3["x"]
            df_3["z_real"] = df_3["z"]
            df_3["y_real"] = df_3["y"]

            if plot_type in ["hist2d", "scatter"]:
                # For the 3D view we use scatter (hist2d is not typical in 3D).
                ax_3d.scatter(df_12["z_real"], df_12["x_real"], df_12["y_real"],
                            marker="o", label="SciFi", alpha=0.7)
                ax_3d.scatter(df_3["z_real"], df_3["x_real"], df_3["y_real"],
                            marker="^", label="Scintillator", alpha=0.7)
            else:
                raise ValueError("plot_type must be either 'hist2d' or 'scatter'")

            ax_3d.set_xlabel("Z [cm]")
            ax_3d.set_ylabel("X [cm]")
            ax_3d.set_zlabel("Y [cm]")
            ax_3d.set_title("3D Event Display. " + f"E_nu = {df_event['E_nu'].iloc[0]:.2f} [GeV]")
            ax_3d.legend()
            # Draw the neutrino arrow in 3D.
            # Compute the neutrino's fiber position for the 3D display.
            tip_z = z_nu
            tip_x = x_nu
            tip_y = y_nu
            scale_3d = 1  # adjust as needed
            delta_z = scale_3d * pz_nu
            delta_x = scale_3d * px_nu
            delta_y = scale_3d * py_nu
            tail_z = tip_z - delta_z
            tail_x = tip_x - delta_x
            tail_y = tip_y - delta_y
            ax_3d.quiver(tail_z, tail_x, tail_y,
                        tip_z - tail_z, tip_x - tail_x, tip_y - tail_y,
                        arrow_length_ratio=0.1, color="red", linewidth=2)
            ax_3d.text(tip_z - 5, tip_x, tip_y + 2, particle_name,
                    color="red", ha="center", va="bottom", fontsize=12)

        plt.tight_layout()
        fig.savefig(f"event_display_digi_{event_id}_{plot_type}_{plot_coords}.pdf")


    def store_events_histograms(self, filename="event_histograms.root", plot_type="hist2d"):
        """
        Process the digitized DataFrame and, for each event, build two 2D numpy histograms:
        - SciFi histogram: hits with layer_type 1 or 2.
        - Scintillator histogram: hits with layer_type 3.
        
        The SciFi histogram uses the digitized layer_id (X axis) and fiber_id_local (Y axis)
        with fixed binning (here we assume 90 layers and self.number_of_fibers fibers).
        
        The Scintillator histogram uses layer_id (X axis) and fiber_id_local (Y axis) with 
        bins corresponding to 45 layers and 100 bins in Y (covering 0 to 2500), and if using
        "hist2d" the hit weights are given by the energy loss (Eloss).

        All events are then stored into a ROOT file (using uproot) as a TTree with branches:
        - event_id  : integer event identifier.
        - sciFi_hist: flattened numpy array holding the SciFi 2D histogram.
        - scint_hist: flattened numpy array holding the scintillator 2D histogram.
        
        These histograms can later be reshaped back to their 2D form for CNN studies.
        
        Parameters:
        filename  (str): The ROOT file name to store the histograms.
        plot_type (str): Either "hist2d" or "scatter" (controls whether scintillator histogram
                        uses weights from Eloss).
        """
        # --- Define fixed binning for the histograms ---
        # For SciFi (digitized) hits:
        total_sciFi_layers = 90  # Assuming 90 layers from 0 to 89
        bins_scifi_x = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
        bins_scifi_y = np.linspace(0, self.number_of_fibers, self.number_of_fibers + 1)
        scifi_shape = (len(bins_scifi_x)-1, len(bins_scifi_y)-1)

        # For Scintillator hits:
        # (Assuming layer_id runs roughly from 0 to 44; adjust binning if needed)
        bins_scint_x = np.arange(-0.5, total_sciFi_layers + 0.5, 1.0)
        bins_scint_y = np.linspace(0, 2500, 2501)
        scint_shape = (len(bins_scint_x)-1, len(bins_scint_y)-1)

        # --- Prepare lists to hold histogram data for each event ---
        event_ids = []
        sciFi_hists = []
        scint_hists = []
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
            # --- Build SciFi histogram ---
            # Filter hits for SciFi (layer_type 1 or 2)
            df_scifi = df_event[df_event["layer_type"].isin([1, 2])]
            if not df_scifi.empty:
                H_scifi, _, _ = np.histogram2d(
                    df_scifi["layer_id"],
                    df_scifi["fiber_id_local"],
                    bins=[bins_scifi_x, bins_scifi_y]
                )
                print("H_scifi: ", H_scifi.shape)
            else:
                H_scifi = np.zeros(scifi_shape)
            # Flatten the histogram to a 1D array for storage.
            sciFi_hists.append(H_scifi.flatten())

            # --- Build Scintillator histogram ---
            # Filter hits for scintillator (layer_type == 3)
            df_scint = df_event[df_event["layer_type"] == 3]
            if not df_scint.empty:
                if plot_type == "hist2d":
                    # Use the energy loss as weight.
                    H_scint, _, _ = np.histogram2d(
                        df_scint["layer_id"],
                        df_scint["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y],
                        weights=df_scint["Eloss"]
                    )
                    print("scint: ", H_scint.shape)
                elif plot_type == "scatter":
                    H_scint, _, _ = np.histogram2d(
                        df_scint["layer_id"],
                        df_scint["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y]
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_scint = np.zeros(scint_shape)
            scint_hists.append(H_scint.flatten())

        # Convert lists to numpy arrays for storage.
        event_ids = np.array(event_ids, dtype=np.int32)
        sciFi_hists = np.array(sciFi_hists, dtype=np.float32)  # shape: (n_events, scifi_flat_length)
        scint_hists = np.array(scint_hists, dtype=np.float32)  # shape: (n_events, scint_flat_length)

        # --- Write the histograms to a ROOT file ---
        # Create a dictionary to be written as a TTree.
        tree_data = {
            "event_id": event_ids,
            "sciFi_hist": sciFi_hists,
            "scint_hist": scint_hists,
            "E_nu": E_nu,
            "theta_nu": theta_nu,
            "nu_flavor": nu_flavor
        }
        
        # Use uproot to create (or overwrite) the ROOT file with a TTree named "EventTree".
        with uproot.recreate(filename) as root_file:
            root_file["EventTree"] = tree_data



    def store_events_3_histograms(self, filename="event_3_histograms.root", plot_type="hist2d"):
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
            df_type1 = df_event[df_event["layer_type"] == 1]
            if not df_type1.empty:
                if plot_type == "hist2d":
                    H_type1, _, _ = np.histogram2d(
                        df_type1["layer_id"],
                        df_type1["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                        weights=df_type1["Eloss"]
                    )
                elif plot_type == "scatter":
                    H_type1, _, _ = np.histogram2d(
                        df_type1["layer_id"],
                        df_type1["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y]
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type1 = np.zeros(scifi_shape)
            hist_type1.append(H_type1.flatten())

            # --- Build histogram for layer_type == 2 ---
            df_type2 = df_event[df_event["layer_type"] == 2]
            if not df_type2.empty:
                if plot_type == "hist2d":
                    H_type2, _, _ = np.histogram2d(
                        df_type2["layer_id"],
                        df_type2["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y],
                        weights=df_type2["Eloss"]
                    )
                elif plot_type == "scatter":
                    H_type2, _, _ = np.histogram2d(
                        df_type2["layer_id"],
                        df_type2["fiber_id_local"],
                        bins=[bins_scifi_x, bins_scifi_y]
                    )
                else:
                    raise ValueError("plot_type must be 'hist2d' or 'scatter'")
            else:
                H_type2 = np.zeros(scifi_shape)
            hist_type2.append(H_type2.flatten())

            # --- Build histogram for layer_type == 3 ---
            df_type3 = df_event[df_event["layer_type"] == 3]
            if not df_type3.empty:
                if plot_type == "hist2d":
                    H_type3, _, _ = np.histogram2d(
                        df_type3["layer_id"],
                        df_type3["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y],
                        weights=df_type3["Eloss"]
                    )
                elif plot_type == "scatter":
                    H_type3, _, _ = np.histogram2d(
                        df_type3["layer_id"],
                        df_type3["fiber_id_local"],
                        bins=[bins_scint_x, bins_scint_y]
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
            "nu_flavor": nu_flavor
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
            event_ids        = tree["event_id"].array(library="np")
            sciFi_hists_flat = tree["sciFi_hist"].array(library="np")
            scint_hists_flat = tree["scint_hist"].array(library="np")
            E_nu             = tree["E_nu"].array(library="np")
            nu_flavor        = tree["nu_flavor"].array(library="np")
        
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
            "nu_flavor": nu_flavor
        }


    def create_fiber_structure(self):
        df_digi = {"event_id": [], "multiplicity": [], "track_id": [], 
                   "E_nu": [], "theta_nu": [], "px_nu": [], "py_nu": [], "pz_nu": [], "x_nu": [], "y_nu": [], "z_nu": [], "pdg_nu": [], 
                   "fiber_id": [], "fiber_id_local": [], "layer_type": [], "layer_id": [], 
                   "x": [], "y": [], "z": [], "pdg": [], "Eloss": []}
        for i, event in enumerate(self.chain):
            E_nu = np.sqrt(event.MCTrack[0].GetPx()**2 + event.MCTrack[0].GetPy()**2 + event.MCTrack[0].GetPz()**2)
            Px_nu, Py_nu, Pz_nu, X_nu, Y_nu, Z_nu = event.MCTrack[0].GetPx(), event.MCTrack[0].GetPy(), event.MCTrack[0].GetPz(), event.MCTrack[0].GetStartX(), event.MCTrack[0].GetStartY(), event.MCTrack[0].GetStartZ()
            event_multiplicity = 0
            for mctrack in event.MCTrack:
                if mctrack.GetMotherId() == 0:
                    event_multiplicity += 1

            if len(event.MTCdetPoint) == 0:
                continue
            for hit in event.MTCdetPoint:
                if hit.GetLayerType() < 3:
                    fiber_id = self.get_global_fiber_id(hit)
                    fiber_id_local = self.get_local_fiber_id(hit)
                    if hit.GetEnergyLoss()*1e6 < self.detector_properties["Eloss_threshold"]:
                        continue
                    # fiber_id_local = fiber_id_local + self.number_of_fibers if hit.GetLayerType() == 2 else fiber_id_local
                else:
                    fiber_id = hit.GetDetectorID()
                    fiber_id_local = fiber_id % 10000




                df_digi["event_id"].append(i)
                df_digi["multiplicity"].append(event_multiplicity)
                df_digi["track_id"].append(hit.GetTrackID())
                df_digi["E_nu"].append(E_nu)
                df_digi["theta_nu"].append(np.arccos(Pz_nu / E_nu)*180/np.pi)
                df_digi["px_nu"].append(Px_nu)
                df_digi["py_nu"].append(Py_nu)
                df_digi["pz_nu"].append(Pz_nu)
                df_digi["x_nu"].append(X_nu)
                df_digi["y_nu"].append(Y_nu)
                df_digi["z_nu"].append(Z_nu)
                df_digi["pdg_nu"].append(event.MCTrack[0].GetPdgCode())
                df_digi["fiber_id"].append(fiber_id)
                df_digi["layer_type"].append(hit.GetLayerType())
                df_digi["fiber_id_local"].append(fiber_id_local)
                # if  hit.GetLayerType() == 3:
                #     df_digi["layer_id"].append(hit.GetLayer() * 2 + 3)
                # else:
                #     df_digi["layer_id"].append(hit.GetLayer() * 2 if hit.GetLayerType() == 1 else hit.GetLayer() * 2 + 1)
                df_digi["layer_id"].append(hit.GetLayer())
                df_digi["x"].append(hit.GetX())
                df_digi["y"].append(hit.GetY())
                df_digi["z"].append(hit.GetZ())
                df_digi["pdg"].append(hit.PdgCode())
                df_digi["Eloss"].append(hit.GetEnergyLoss())
        df_digi = pd.DataFrame(df_digi)

        self.df_digi = df_digi
                



import argparse
import os
import glob

def main():
    parser = argparse.ArgumentParser(description="Analyze FairShip data")
    parser.add_argument("--fiber_pitch", type=float, default=0.1, help="Fiber pitch")
    parser.add_argument("--width", type=float, default=50, help="Detector width")
    parser.add_argument("--height", type=float, default=50, help="Detector height")
    parser.add_argument("--startZ", type=float, default=-3511.7150, help="Start Z coordinate")
    parser.add_argument("--endZ", type=float, default=-3175.5650, help="End Z coordinate")
    parser.add_argument("--angle", type=float, default=5, help="Detector tilt angle")
    parser.add_argument("--Eloss_threshold", type=float, default=180, help="Energy loss threshold")
    parser.add_argument("--event_id", type=int, default=22, help="Event ID to display")
    parser.add_argument("--plot_type", type=str, default="hist2d", choices=["hist2d", "scatter"], help="Type of plot")
    parser.add_argument("--plot_coords", type=str, default="both", choices=["digi", "zx", "both", "3d"], help="Coordinate system(s) to plot")
    parser.add_argument("--mode", type=str, default="dummy", choices=["display", "store", "read", "dummy"], help="Mode of operation")
    parser.add_argument("--input", type=str, default="ship.conical.Genie-TGeant4.root", help="Input filename (used in single mode)")
    parser.add_argument("--output", type=str, default="event_histograms.root", help="Output filename (used in single mode)")
    parser.add_argument("--mode_batch", type=str, default="single", choices=["single", "batch"], help="Run on one file or batch of 1–1000")
    parser.add_argument("--batch_flavor", type=str, default="14", choices=["12", "14", "16"], help="Choose flavor")
    args = parser.parse_args()

    if args.mode_batch == "batch":
        pattern = f"/eos/experiment/ship/user/edursov/pycondor_out/CCDIS_100k_1/{args.batch_flavor}/*/ship.conical.Genie-TGeant4.root"
        input_files = sorted(glob.glob(pattern))
        if not input_files:
            print(f"No files found matching pattern: {pattern}")
        for input_path in input_files:
            dir_path = os.path.dirname(input_path)
            output_path = os.path.join(dir_path, "event_3_histograms_pr-signal.root")
            if not os.path.isfile(input_path):
                print(f"Skipping: {input_path} not found.")
                continue
            print(f"Processing {input_path}")
            run_single(input_path, output_path, args)
        else:
            run_single(args.input, args.output, args)

def run_single(input_file, output_file, args):
    fiber_dimensions = {
        "fiber_pitch": args.fiber_pitch,
    }
    detector_properties = {
        "width": args.width,
        "height": args.height,
        "startZ": args.startZ,
        "endZ": args.endZ,
        "angle": args.angle,
        "Eloss_threshold": args.Eloss_threshold
    }

    analyzer = FairShipAnalyzer(input_file, detector_properties, fiber_dimensions)
    if analyzer.chain is None:
        print(f"Error: Unable to open file {input_file}.")
        return 
    analyzer.create_fiber_structure()

    if args.mode == "display":
        analyzer.event_display_digi_new(args.event_id, args.plot_type, args.plot_coords)
    elif args.mode == "store":
        analyzer.store_events_3_histograms(filename=output_file, plot_type=args.plot_type)
    elif args.mode == "read":
        analyzer.read_events_histograms(filename=output_file)
    else:
        analyzer.analyze_events()


if __name__ == "__main__":
    main()


