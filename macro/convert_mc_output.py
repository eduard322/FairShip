import ROOT as r
import pandas as pd
import numpy as np
from argparse import ArgumentParser

branches = [
        "Ev",      # incoming neutrino energy
        "pxv",     # x-component of incoming neutrino momentum
        "pyv",     # y-component of incoming neutrino momentum
        "pzv",     # z-component of incoming neutrino momentum
        "neu",     # incoming neutrino PDG code
        "cc",      # Is it a CC event?
        "nuel",    # Is it a NUEEL event?
        "vtxx",    # vertex x-coordinate in SI units
        "vtxy",    # vertex y-coordinate in SI units
        "vtxz",    # vertex z-coordinate in SI units
        "vtxt",    # vertex time in SI units
        "El",      # outgoing lepton energy
        "pxl",     # x-component of outgoing lepton momentum
        "pyl",     # y-component of outgoing lepton momentum
        "pzl",     # z-component of outgoing lepton momentum
        "Ef",      # outgoing hadronic momenta
        "pxf",     # x-component of outgoing hadronic momenta
        "pyf",     # y-component of outgoing hadronic momenta
        "pzf",     # z-component of outgoing hadronic momenta
        "nf",      # number of outgoing hadrons
        "pdgf"     # PDG code of hadrons
    ]

def read_mc_output(file_mc):
    FILE = r.TFile.Open(file_mc)
    tree = FILE.Get("cbmsim")
    int_points = {"x": [], "y": [], "z": []}
    for event in tree:
        # print(event.MCTrack[0].GetStartX(), event.MCTrack[0].GetStartY(), event.MCTrack[0].GetStartZ(), event.MCTrack[0].GetPdgCode())
        int_points["x"].append(event.MCTrack[0].GetStartX())
        int_points["y"].append(event.MCTrack[0].GetStartY())
        int_points["z"].append(event.MCTrack[0].GetStartZ())
    return int_points



def read_genie_output(file_genie, first_event = 0, event_num = 100):
    FILE = r.TFile.Open(file_genie)
    tree = FILE.Get("gst")

    points = {"Event": []}
    for i in range(first_event, first_event + event_num):
        tree.GetEntry(i)
        points["Event"].append(i)
        for branch in branches:
            if branch not in points:
                points[branch] = []
            if "f" in branch and branch != "nf":
               points[branch].append([x for x in eval(f"tree.{branch}")]) 
            else:
                points[branch].append(eval(f"tree.{branch}"))

    return points

def merge(genie_points, int_points):
    genie_points["x"] = int_points["x"][:]
    genie_points["y"] = int_points["y"][:]
    genie_points["z"] = int_points["z"][:]
    genie_points_exploded = {key: [] for key in genie_points}
    for i in range(len(genie_points["Ev"])):
        for Ef, pxf, pyf, pzf, pdgf in zip(genie_points["Ef"][i], genie_points["pxf"][i], genie_points["pyf"][i], genie_points["pzf"][i], genie_points["pdgf"][i]):
            genie_points_exploded["Ef"].append(Ef)
            genie_points_exploded["pxf"].append(pxf)
            genie_points_exploded["pyf"].append(pyf)
            genie_points_exploded["pzf"].append(pzf)
            genie_points_exploded["pdgf"].append(pdgf)
            for branch in genie_points:
                if "f" not in branch or branch == "nf":
                     genie_points_exploded[branch].append(genie_points[branch][i])

    genie_points = pd.DataFrame(genie_points)
    #print(genie_points_exploded.keys())
    #print([len(genie_points_exploded[key]) for key in genie_points_exploded])
    genie_points_exploded = pd.DataFrame(genie_points_exploded)
    #print(genie_points[["pyf", "pzf", "pdgf"]])
    #print(genie_points_exploded[["pyf", "pzf", "pdgf"]])
    genie_points_exploded = genie_points_exploded.query("abs(x) < 25.")
    genie_points_exploded.to_csv("output_sampled.csv")


parser = ArgumentParser()
parser.add_argument("-g", "--genie",dest="genie",  help="genie file", required=False,  default="/eos/experiment/ship/user/edursov/data_3_2/genie-nu_tau_full_new.root")
parser.add_argument("-m", "--mc",dest="mc",  help="mc file", required=False,  default="ship.conical.Genie-TGeant4.root")
options = parser.parse_args()


mc_data = read_mc_output(options.mc)
genie_data = read_genie_output(options.genie, 0, len(mc_data["x"]))

merge(genie_data, mc_data)
# read_mc_output("ship.conical.Genie-TGeant4.root")
