#!/bin/bash

# edit 20250805
# Input of pgun should be from 0722; source is changed.
# echo "Hits"
SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250805_scale4to4p522/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5"
# SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5"
HitFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_tred_hits.hdf5"
HOFile="pgun_mu_test_hits.hdf5"

EffQFile=${HitFile/_hits.hdf5/_effq.hdf5}
QOFile=${HOFile/_hits.hdf5/_effq.hdf5}

echo "Hits"
uv run ../pick_pgun.py $SrcFile $HitFile $HOFile
# echo "EffQ"
uv run ../pick_pgun.py --dtype=effq $SrcFile $EffQFile $QOFile
