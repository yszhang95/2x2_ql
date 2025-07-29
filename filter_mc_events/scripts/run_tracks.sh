#!/bin/bash
# skip # in the file
# edit 20250718
# event_ids=$(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250708/event_list.txt) # works
# edit 20250722
# event_ids=$(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/event_list.txt) # works
# edit 20250724; variants of 0722, adc_hold_delay=1.8us
# event_ids=$(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/event_list.txt) # works
# edit 20250728; variants of 0722, csa_reset_time=0us
event_ids=$(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/event_list.txt) # works
for i in ${event_ids[@]}; do
   # skip # in the file
   if [[ $i == \#* ]]; then
       echo "Skipped comment line: $i"
       continue
   fi
   # edit 20250718
   # InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_event_id${i}_tred.npz"
   # edit 20250722
   # InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}_tred.npz"
   # edit 20250724
   # InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}_delay18_tred.npz"
   # edit 20250728; no reset time
   InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}_noreset_tred.npz"
#    From =.npz= to =hdf5=, a format more compatatible with existing ND,
# : uv run python many_muons.py /path/to/npz /path/to/output [hits|effq]
# A =many_muons.hdf5= will be created in ~/path/to/output/~.
   OutDir=$(dirname ${InFile})
   echo "Converting $InFile to directory $OutDir"
   uv run python ../many_muons.py  $InFile $OutDir hits
   uv run python ../many_muons.py  $InFile $OutDir effq

done
