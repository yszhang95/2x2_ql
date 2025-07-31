#!/bin/bash

# edit 20250718
# echo "Hits"
# uv run ../pick_pgun.py /home/yousen/Public/ndlar_shared/data_reflowv5_20250708/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_event_id{}_tred_hits.hdf5" /home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_merged_hits.hdf5
# echo "EffQ"
# uv run ../pick_pgun.py --dtype=effq /home/yousen/Public/ndlar_shared/data_reflowv5_20250708/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_event_id{}_tred_effq.hdf5" /home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_merged_effq.hdf5

# edit 20250722
# echo "Hits"
# uv run ../pick_pgun.py /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_tred_hits.hdf5" /home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_merged_hits.hdf5
# echo "EffQ"
# uv run ../pick_pgun.py --dtype=effq /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_tred_effq.hdf5" /home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_merged_effq.hdf5

# edit 20250724
# same events as in 0722 because they are from variants of TRED setup, by changing ADC_HOLD_DELAY
# echo "Hits"
# uv run ../pick_pgun.py --no_sel /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_delay18_tred_hits.hdf5" /home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_merged_hits.hdf5
# echo "EffQ"
# uv run ../pick_pgun.py --dtype=effq /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_delay18_tred_effq.hdf5" /home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_merged_effq.hdf5


# edit 20250728
# same events as in 0722 because they are from variants of TRED setup, by changing csa_reset_time == 0
# SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "
# HitFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_delay18_noreset_tred_hits.hdf5"
# HOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_noreset_merged_hits.hdf5"
# EffQFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_delay18_noreset_tred_effq.hdf5"
# QOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_noreset_merged_effq.hdf5"

# edit 20250728; delay 18 + no reset
# SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "
# HitFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_delay18_noreset_tred_hits.hdf5"
# HOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_noreset_merged_hits.hdf5"
# EffQFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id{}_delay18_noreset_tred_effq.hdf5"
# QOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_delay18_noreset_merged_effq.hdf5"

# edit 20250730; xoffset 0.5cm to anode
# SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5 "
# HitFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_event_id{}_tred_hits.hdf5"
# HOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_merged_hits.hdf5"
# EffQFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_event_id{}_tred_effq.hdf5"
# QOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_merged_effq.hdf5"

# edit 20250730; xoffset 1cm to anode
# SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5"
# HitFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_1p0cm_event_id{}_tred_hits.hdf5"
# HOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_1p0cm_merged_hits.hdf5"
# edit 20250730; xoffset 2cm to anode
SrcFile="/home/yousen/Public/ndlar_shared/data_reflowv5_20250722/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5"
HitFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_2p0cm_event_id{}_tred_hits.hdf5"
HOFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_2p0cm_merged_hits.hdf5"

EffQFile=${HitFile/_hits.hdf5/_effq.hdf5}
QOFile=${HOFile/_hits.hdf5/_effq.hdf5}

echo "Hits"
uv run ../pick_pgun.py --no_sel $SrcFile $HitFile $HOFile
echo "EffQ"
uv run ../pick_pgun.py --dtype=effq $SrcFile $EffQFile $QOFile
