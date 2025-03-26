
# 
INPUT=/eos/experiment/ship/user/edursov/data_3_2/genie-nu_tau_full_new.root
python $FAIRSHIP/macro/run_simScript.py --Genie -n 100 -f $INPUT --noSND --noSC -g combi.root --seed 1
