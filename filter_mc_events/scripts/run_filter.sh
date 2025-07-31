#!/bin/bash

# edit 20250724
# uv run python filter_hdf5.py -o pgun_mu_3GeV_2mm_20250724_delay18_filtered_hits.hdf5 \
#    "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_merged_hits.hdf5" ../selected_event_run_ids.txt

# edit 20250728; no reset time
# OFile="pgun_mu_3GeV_2mm_20250728_noreset_filtered_hits.hdf5"
# OFileGroup=$(echo $OFile | sed 's/_hits\.hdf5/_*.hdf5/')
# uv run python ../pgun_display/filter_hdf5.py -o pgun_mu_3GeV_2mm_20250728_noreset_filtered_hits.hdf5 \
#    "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_noreset_merged_hits.hdf5" ../pgun_display/selected_event_run_ids.txt
# mv $OFileGroup ../pgun_mu_20250728

# edit 20250728; delay 18 + no reset
# IFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_noreset_merged_hits.hdf5"
# OFile="pgun_mu_3GeV_2mm_20250728_delay18_noreset_filtered_hits.hdf5"
# OFileGroup=$(echo $OFile | sed 's/_hits\.hdf5/_*.hdf5/')

# edit 20250730; xoffset 0.5cm to anode
# IFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_merged_hits.hdf5"
# OFile="pgun_mu_3GeV_2mm_20250730_xoffset_0p5cm_filtered_hits.hdf5"

# edit 20250730; xoffset 1cm to anode
# IFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_1p0cm_merged_hits.hdf5"
# OFile="../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_1p0cm_filtered_hits.hdf5"

# edit 20250730; xoffset 2cm to anode
IFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_2p0cm_merged_hits.hdf5"
OFile="../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_2p0cm_filtered_hits.hdf5"


OFileGroup=$(echo $OFile | sed 's/_hits\.hdf5/_*.hdf5/')

uv run python ../pgun_display/filter_hdf5.py -o $OFile \
   $IFile \
   ../pgun_display/selected_event_run_ids.txt

# edit 20250728
# mv $OFileGroup ../pgun_mu_20250728

# edit 20250730
# mv $OFileGroup ../pgun_mu_20250730
