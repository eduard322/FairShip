#!/usr/bin/env python
from __future__ import annotations
import ROOT,os,sys,time
from subprocess import call
import shipunit as u
import shipRoot_conf
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import genie_interface
shipRoot_conf.configure()

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

# IMPORTANT
# Before runnig this script please run this command in FairShip bash if you are dealing with the neutrino detector:
# export GXMLPATH='/eos/experiment/ship/user/aiuliano/GENIE_FNAL_nu_splines'
# this will disable Genie decays for charm particles and tau



xsec = "gxspl-FNALsmall.xml"# new adapted splines from Genie site
hfile = "pythia8_Geant4_1.0_withCharm_nu.root" #2018 background generation
#xsec = "Nu_splines.xml"
#hfile = "pythia8_Geant4-withCharm_onlyNeutrinos.root"


defaultsplinedir   = '/eos/experiment/ship/user/edursov/genie/genie_xsec/v3_02_00/NULL/G1802a00000-k250-e1000/data' #path of splines
defaultfiledir  = '/eos/experiment/ship/data/Mbias/background-prod-2018' #path of flux


neutrino_code_mapping = {x:  "10" + str(x) for x in [12, 14, 16]} | {x: "20" + str(-x) for x in [-12, -14, -16]}
target_code_mapping = {
    "iron": "1000260560",
    "lead": "1000822040[0.014],1000822060[0.241],1000822070[0.221],1000822080[0.524]",
    "tungsten": "1000741840"
}


def generate_genie_events(
    nevents: int,
    nupdg: int,
    emin: float,
    emax: float,
    targetcode: str,
    inputflux: str,
    spline: str,
    outputfile: str,
    process: Optional[str] = None,
    seed: Optional[int] = None,
    irun: Optional[int] = None,
    *,
    prod_mode: bool = True,                 # sets GPRODMODE=YES
    messenger: Optional[str] = "Messenger_laconic.xml",  # verbosity config
    capture_output: bool = True,            # capture stdout/stderr (keeps console clean)
    devnull_output: bool = False,           # discard output entirely (overrides capture_output)
    timeout: Optional[int] = None           # seconds
) -> subprocess.CompletedProcess:
    """
    Launch GENIE gevgen with robust defaults and quiet logging.
    Raises subprocess.CalledProcessError if gevgen fails.
    Returns the CompletedProcess for inspection.
    """

    # --- quick sanity checks -----------------------------------------------------------
    if nevents <= 0:
        raise ValueError("nevents must be > 0")
    if emin < 0 or emax <= emin:
        raise ValueError("Require 0 <= emin < emax")
    if not targetcode:
        raise ValueError("targetcode must be a non-empty GENIE nuclear code")

    gevgen_path = shutil.which("gevgen")
    if not gevgen_path:
        raise RuntimeError("Cannot find 'gevgen' in PATH.")

    # flux and spline may be on EOS/CVMFS; just warn if not a plain file path
    # (gevgen will still error clearly if paths are wrong)
    for pth, label in [(inputflux, "inputflux"), (spline, "spline")]:
        if ("," in pth) and label == "inputflux":
            # ok: gevgen expects "-f flux.root,FluxDriverName"
            pass
        else:
            # allow non-local schemes; only warn if it *looks* like a local path and missing
            p = Path(pth)
            if p.anchor and not p.exists():
                print(f"[generate_genie_events] Warning: {label} path not found: {pth}")

    # --- message thresholds file -------------------------------------------------------
    msg_arg: Sequence[str] = []
    if messenger:
        # Accept bare filename or absolute/expanded path. If user passed something like
        # "$GENIE/config/Messenger_laconic.xml", expand env vars here.
        expanded = os.path.expandvars(messenger)
        if not expanded.endswith(".xml"):
            # user might have passed just a base like 'Messenger_laconic'
            expanded += ".xml"

        if not os.path.isabs(expanded):
            # try to resolve via $GENIE/config if available
            genie_root = os.environ.get("GENIE")
            if genie_root:
                candidate = Path(genie_root) / "config" / expanded
                expanded = str(candidate)

        msg_arg = ["--message-thresholds", expanded]

    # --- build argv (no shell=True) ---------------------------------------------------
    # -f takes "fluxfile,DriverName". Caller is giving that as `inputflux`.
    argv = [
        gevgen_path,
        "-n", str(nevents),
        "-p", str(nupdg),
        "-t", str(targetcode),
        "-e", f"{emin},{emax}",
        *msg_arg,
        "-f", inputflux,
        "--cross-sections", spline,
        "-o", outputfile,
    ]
    if process:
        argv += ["--event-generator-list", process]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if irun is not None:
        argv += ["--run", str(irun)]

    # --- environment: set GPRODMODE only for this process -----------------------------
    env = os.environ.copy()
    if prod_mode:
        env["GPRODMODE"] = "YES"   # quiet: thresholds bumped to WARNING in GENIE

    # --- output handling ---------------------------------------------------------------
    stdout = stderr = None
    if devnull_output:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    elif capture_output:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    # --- run --------------------------------------------------------------------------
    print("Starting GENIE with argv:\n  " + shlex.join(argv))
    try:
        cp = subprocess.run(
            argv,
            env=env,
            check=True,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
        )
        return cp
    except subprocess.CalledProcessError as e:
        # Bubble up detailed message for debugging if captured
        msg = f"gevgen failed with code {e.returncode}"
        if e.stdout:
            msg += f"\n--- stdout ---\n{e.stdout}"
        if e.stderr:
            msg += f"\n--- stderr ---\n{e.stderr}"
        raise RuntimeError(msg) from e




def get_arguments(): #available options

  parser = argparse.ArgumentParser(
      description='Run GENIE neutrino" simulation')
  subparsers = parser.add_subparsers()
  ap = subparsers.add_parser('sim',help="make genie simulation file")

  ap.add_argument('-s', '--seed', type=int, dest='seed', default=65539) #default seed in $GENIE/src/Conventions/Controls.h
  ap.add_argument('-o','--output'    , type=str, help="output directory", dest='work_dir', default=None)
  ap.add_argument('-f','--filedir', type=str, help="directory with neutrino fluxes", dest='filedir', default=defaultfiledir)
  ap.add_argument('-c','--crosssectiondir', type=str, help="directory with neutrino splines crosssection", dest='splinedir', default=defaultsplinedir)
  ap.add_argument('-t', '--target', type=str, help="target material", dest='target', default='iron')
  ap.add_argument('-n', '--nevents', type=int, help="number of events", dest='nevents', default=100)
  ap.add_argument('-e', '--event-generator-list', type=str, help="event generator list", dest='evtype', default=None) # Possbile evtypes: CC, CCDIS, CCQE, CharmCCDIS, RES, CCRES, see other evtypes in $GENIE/config/EventGeneratorListAssembler.xml
  ap.add_argument("--nudet", dest="nudet", help="option for neutrino detector", required=False, action="store_true")
  ap.add_argument('-p','--particles', dest ="particles", nargs="+", type = int, help="particles", default=16)
  ap.add_argument('-r', '--run', type=int, help="run number", dest="run", default=1)   # <-- added here
  ap1 = subparsers.add_parser('spline',help="make a new cross section spline file")
  ap1.add_argument('-t', '--target', type=str, help="target material", dest='target', default='iron')
  ap1.add_argument('-o','--output'    , type=str, help="output directory", dest='work_dir', default=None)
  args = parser.parse_args()
  return args


def extract_nu_over_nubar(neutrino_flux, particles):
    """Extract the ratio of neutrino to antineutrino events from the flux file."""
    nuOverNubar = {}
    f = ROOT.TFile(neutrino_flux)
    for x in particles:
        print(f"Extracting {neutrino_code_mapping[x]} from {neutrino_flux}")
        nuOverNubar[x] = f.Get(neutrino_code_mapping[x]).GetSumOfWeights() / f.Get(neutrino_code_mapping[-x]).GetSumOfWeights()
    f.Close()
    return nuOverNubar

def makeSplines():
 '''first step, make cross section splines if not exist'''
 nupdglist = [16,-16,14,-14,12,-12]
 genie_interface.make_splines(nupdglist, targetcode, 400, nknots = 500, outputfile = "xsec_splines.xml")

def makeEvents(run = 0, nevents = 100, particles = [16], targetcode = '1000260560', process = "CC", emin = 0.5, emax = 350, neutrino_flux = None, splines = None, seed = 42, work_dir = None):
    pdg  = ROOT.TDatabasePDG()
    for i, x in enumerate(particles):
        pdg_name = pdg.GetParticle(x).GetName()
        if x < 0:
            print(f"Scale number of {pdg_name} events with {1./nuOverNubar[abs(x)]:.2f}")

        # stop at 350 GeV, otherwise strange warning about "Lower energy neutrinos have a higher probability of
        # interacting than those at higher energy. pmaxLow(E=386.715)=2.157e-13 and  pmaxHigh(E=388.044)=2.15623e-13"
        N = nevents if x > 0 else int(nevents / nuOverNubar[abs(x)])

        output_dir = os.path.join(work_dir, f"genie-{pdg_name}_{N}_events")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"run_{run + i}_{pdg_name}_{N}_events_{targetcode}_{emin}_{emax}_GeV_{process}.ghep.root"
        # os.chdir(output_dir)

        genie_interface.generate_genie_events(nevents = N, nupdg = x, targetcode = targetcode, emin = emin, emax = emax,
                      inputflux = neutrino_flux, spline = splines, seed = seed, process = process, irun = run + i, outputfile = os.path.join(output_dir, filename))
        # generate_genie_events(nevents = N, nupdg = x, targetcode = targetcode, emin = emin, emax = emax,
        #               inputflux = neutrino_flux, spline = splines, seed = seed, process = process, irun = run + i, outputfile = os.path.join(output_dir, filename))
        genie_interface.make_ntuples(os.path.join(output_dir, filename), os.path.join(output_dir, f"genie-{filename}"))
        genie_interface.add_hists(neutrino_flux, os.path.join(output_dir, f"genie-{filename}"), x)

# def makeEvents_MP(run = 0, nevents = 100, particles = [16], targetcode = '1000260560', process = "CC", emin = 0.5, emax = 350, neutrino_flux = None, splines = None, seed = 42):
#     pdg  = ROOT.TDatabasePDG()
#     for i, x in enumerate(particles):
#         pdg_name = pdg.GetParticle(x).GetName()
#         if x < 0:
#             print(f"Scale number of {pdg_name} events with {1./nuOverNubar[abs(x)]:.2f}")

#         # stop at 350 GeV, otherwise strange warning about "Lower energy neutrinos have a higher probability of
#         # interacting than those at higher energy. pmaxLow(E=386.715)=2.157e-13 and  pmaxHigh(E=388.044)=2.15623e-13"
#         N = nevents if x > 0 else int(nevents / nuOverNubar[abs(x)])

#         output_dir = os.path.join(args.work_dir, f"genie-{pdg_name}")
#         os.makedirs(output_dir, exist_ok=True)
#         filename = f"run_{run + i}_{pdg_name}_{N}_events_{targetcode}_{emin}_{emax}_GeV_{process}.ghep.root"
#         # os.chdir(output_dir)

#         genie_interface.generate_genie_events(nevents = N, nupdg = x, targetcode = targetcode, emin = emin, emax = emax,
#                       inputflux = neutrino_flux, spline = splines, seed = seed, process = process, irun = run + i, outputfile = os.path.join(output_dir, filename))
#         genie_interface.make_ntuples(os.path.join(output_dir, filename), os.path.join(output_dir, f"genie-{filename}.root"))
#         genie_interface.add_hists(neutrino_flux, os.path.join(output_dir, f"genie-{filename}.root"), x)



if __name__ == "__main__":
    args = get_arguments()
    if args.nudet:
        print("Neutrino detector option is enabled. Charm and tau decays will be disabled.")
    else:
        print("Neutrino detector option is disabled. Charm and tau decays will be enabled.")

    print(f"Target type: {args.target}")

    targetcode = target_code_mapping.get(args.target, None)
    if targetcode is None:
        print('Only iron, lead and tungsten target options available')
        sys.exit(1)
    os.makedirs(args.work_dir, exist_ok=True)
    print("Starting the GENIE event generation process...")


    splines = os.path.join(args.splinedir, xsec)  #path of splines
    neutrino_flux = os.path.join(args.filedir, hfile) #path of flux

    print(f"Seed used in this generation: {args.seed}")
    print(f"Splines file used {xsec}")

    nuOverNubar = {}
    f = ROOT.TFile(neutrino_flux)
    particles = args.particles if isinstance(args.particles, list) else [args.particles]

    nuOverNubar = extract_nu_over_nubar(neutrino_flux, particles)
    print(args.nevents)
    makeEvents(run = args.run, nevents = args.nevents, particles = particles, targetcode = targetcode, process = args.evtype,
               emin = 0.5, emax = 350, neutrino_flux = neutrino_flux, splines = splines, seed = args.seed, work_dir= args.work_dir)

    print("Event generation completed successfully.")