#!/bin/bash

# Load FairShip
year=25
month=09
echo Loading FairSHiP release ${year}.${month}
source /cvmfs/ship.cern.ch/${year}.${month}/setUp.sh
git lfs install

# Verify installations
echo $SHIPDIST
echo $ALIBUILD
echo $FAIRSHIP
which python
python --version
which aliBuild

# -------------------------------------------------------------------------------------------------------------------
#                                     Useful commands
# -------------------------------------------------------------------------------------------------------------------

# To build with aliBuild:
# aliBuild build FairShip --architecture slc9_x86-64 --always-prefer-system --config-dir $SHIPDIST --defaults release
# The build will be saved in /sw/

# To load env later, after sourcing the setUp.sh, run:
# alienv enter FairShip/latest-release

# To load env manually (careful with the release name), change to where you have compiled
# export WORK_DIR=/afs/cern.ch/user/m/marquezh/public/SHiP/sw
# source $WORK_DIR/slc9_x86-64/FairShip/latest-square_ms_snd-release/etc/profile.d/init.sh

# In VM:
# export WORK_DIR=/home/almalinux/public/SHiP/sw
