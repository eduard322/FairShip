# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright CERN for the benefit of the SHiP Collaboration

import global_variables
import ROOT
import SciFiMapping
from BaseDetector import BaseDetector


class MTCDetector(BaseDetector):
    def __init__(self, name, intree, branchName=None, outtree=None) -> None:
        super().__init__(name, intree, branchName, outtree=outtree)
        # add MTC module to the list of globals to use it later in the MTCDetHit class. Consistent with SND@LHC approach.
        # make SiPM to fibre mapping
        if intree.GetBranch("MTCDetPoint"):
            lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
            if not lsOfGlobals.FindObject("MTC"):
                lsOfGlobals.Add(global_variables.modules["MTC"])
            mapping = SciFiMapping.SciFiMapping(global_variables.modules)
            mapping.make_mapping()
            # Per-block SiPM maps: the MTC is split longitudinally into blocks of
            # different transverse size, so each block has its own map. These are
            # lists indexed by block; select with block = layer // n_layers_per_block.
            self.sipm_to_fibre_map_U, self.sipm_to_fibre_map_V = mapping.get_sipm_to_fibre_map()
            mtc = global_variables.modules["MTC"]
            self.n_blocks = mtc.GetNBlocks()
            self.n_layers_per_block = mtc.GetNLayersPerBlock()

    def digitize(self) -> None:
        """Digitize SND/MTC MC hits.

        Example of fiberID: 123051820, where:
          - 1: MTC unique ID
          - 23: layer number
          - 0: station type (0 for +5 degrees, 1 for -5 degrees, 2 for scint plane)
          - 5: z-layer number (0-5)
          - 1820: local fibre ID within the station
        Example of SiPM global channel (what is seen in the output file): 123001123, where:
          - 1: MTC unique ID
          - 23: layer number
          - 0: station type (0 for +5 degrees, 1 for -5 degrees)
          - 0: mat number (only 0 by June 2025). In future, if multiple mats per station are used,
          this number will be 0-N. Currently, this digit is allocated by SiPM channel ID!
          - 1: SiPM number (automatically assigned based on fibre aggregation settings)
          - 123: number of the SiPM channel (0-N). The channel number depends on the fibre aggregation setting.
        """
        hit_container = {}
        mc_points = {}
        norm = {}
        for k, mc_point in enumerate(self.intree.MTCDetPoint):
            det_id = mc_point.GetDetectorID()
            # 0: +5 deg, 1: -5 deg, 2: scint plane
            station_type = mc_point.GetLayerType()
            energy_loss = mc_point.GetEnergyLoss()

            # Which longitudinal block this hit belongs to (blocks differ in
            # transverse size, hence in their SiPM map).
            layer = (det_id // 1_000_000) % 100
            block = min(layer // self.n_layers_per_block, self.n_blocks - 1)

            if station_type == 0:
                # +5 degrees fiber station uses U fibers
                fibre_map = self.sipm_to_fibre_map_U[block]
            elif station_type == 1:
                # -5 degrees fiber station uses V fibers
                fibre_map = self.sipm_to_fibre_map_V[block]
            elif station_type == 2:
                # Scint Plane. Preserve the same logic as for fibre stations,
                # but use the det_id directly as the global channel.
                global_channel = det_id
                if global_channel not in hit_container:
                    hit_container[global_channel] = []
                    mc_points[global_channel] = {}
                    norm[global_channel] = 0

                weight = 1
                # Append (energy_loss, weight) instead of (mc_point, weight)
                hit_container[global_channel].append([mc_point, weight])
                d_e = energy_loss * weight
                mc_points[global_channel][k] = d_e
                norm[global_channel] += d_e
                continue
            else:
                # Skip any other station_type values
                continue
            # For station_type 0 or 1, look up the local fibre ID
            loc_fibre_id = det_id % 1_000_000
            if loc_fibre_id not in fibre_map:
                # If there is no entry for this fibre ID, skip
                print(
                    f"MTC digitization: no mapping found for fibre ID {loc_fibre_id} in station type {station_type}. Skipping."
                )
                continue

            for sipm_chan, chan_info in fibre_map[loc_fibre_id].items():
                global_channel = (det_id // 1_000_000) * 1_000_000 + sipm_chan
                if global_channel not in hit_container:
                    hit_container[global_channel] = []
                    mc_points[global_channel] = {}
                    norm[global_channel] = 0

                weight = chan_info["weight"]
                hit_container[global_channel].append([mc_point, weight])
                d_e = energy_loss * weight
                mc_points[global_channel][k] = d_e
                norm[global_channel] += d_e

        for det_id in hit_container:
            all_points = ROOT.std.vector("MTCDetPoint*")()
            all_weights = ROOT.std.vector("Float_t")()

            for entry in hit_container[det_id]:
                all_points.push_back(entry[0])
                all_weights.push_back(entry[1])
            det_hit = ROOT.MTCDetHit(det_id, all_points, all_weights)
            self.det.push_back(det_hit)
            # digi2MCPoints will be added later
            #   for idx, de_value in mc_points[det_id].items():
            #     mc_links.Add(det_id, idx, de_value / norm[det_id])
