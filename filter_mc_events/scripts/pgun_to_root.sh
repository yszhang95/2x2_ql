#!/bin/bash

# edit 20250724; delay 18
# uv run ../asymmetry_prep2.py \
#    ../pgun_display/pgun_mu_3GeV_2mm_20250724_delay18_filtered_hits.hdf5 \
#    ../threshold_summary.hdf5
# mv ../pgun_display/pgun_mu_3GeV_2mm_20250724_delay18_filtered_hits.hdf5  \
#    ../pgun_mu_20250724/
#
# uv run ../project_effq.py \
#    ../pgun_display/pgun_mu_3GeV_2mm_20250724_delay18_filtered_effq.hdf5 \
#    0 1
# mv ../pgun_display/pgun_mu_3GeV_2mm_20250724_delay18_filtered_effq.hdf5  \
#    ../pgun_mu_20250724/

# edit 20250722; default setup; bug fix
HitInFile="../pgun_mu_20250722/pgun_mu_3GeV_2mm_20250722_filtered_hits.hdf5"

# edit 20250724; delay 18; bug fix
# HitInFile="../pgun_mu_20250724/pgun_mu_3GeV_2mm_20250724_delay18_filtered_hits.hdf5"
# QInFile="../pgun_mu_20250724/pgun_mu_3GeV_2mm_20250724_delay18_filtered_effq.hdf5"

# edit 20250728; no reset
# HitInFile="../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_noreset_filtered_hits.hdf5"
# QInFile="../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_noreset_filtered_effq.hdf5"

# edit 20250728; delay 18 no reset
# HitInFile="../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_delay18_noreset_filtered_hits.hdf5"
# QInFile="../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_delay18_noreset_filtered_effq.hdf5"

# edit 20250730; xoffset 0.5cm to anode
# HitInFile="../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_0p5cm_filtered_hits.hdf5"

# edit 20250730; xoffset 1cm to anode
# HitInFile="../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_1p0cm_filtered_hits.hdf5"

# edit 20250730; xoffset 2cm to anode
# HitInFile="../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_2p0cm_filtered_hits.hdf5"

# replace _hits with _effq in bash
QInFile="${HitInFile/_hits.hdf5/_effq.hdf5}"

uv run ../asymmetry_prep2.py $HitInFile ../threshold_summary.hdf5
uv run ../project_effq.py $QInFile 0 1
