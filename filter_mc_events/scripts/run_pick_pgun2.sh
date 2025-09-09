#!/bin/bash

while IFS= read -r InFile; do
    echo "Processing $InFile"
    SrcFile="$InFile"
    bname=$(basename "$InFile")
    odir="${bname%.hdf5}"

    SrcBase=$(basename "$SrcFile")
    HitFileDir="${SrcBase%.hdf5}"
    HitFile="/home/yousen/Documents/NDLAr2x2/tred/tests/playground/for_batch_pgun/${HitFileDir}/pgun_mu_3GeV_2mm.npz"
    HOFile="${HitFileDir}/pgun_mu_3GeV_2mm_20250908.hdf5"

    if [[ ! -d $HitFileDir ]]; then
        echo "Output directory $HitFileDir does not exist. Creating it."
        echo $HitFileDir
        mkdir "$HitFileDir"
    fi

    if [[ ! -f $HOFile ]]; then
        echo "Hits hdf5 file $HOFile does not exists. Skipping."
        continue
    fi

    # EffQFile=${HitFile/_hits.hdf5/_effq.hdf5}
    # QOFile=${HOFile/_hits.hdf5/_effq.hdf5}
    #
    # echo "Hits"
    # uv run ../pick_pgun2.py $SrcFile $HitFile $HOFile
    # # echo "EffQ"
    # uv run ../pick_pgun2.py --dtype=effq $SrcFile $EffQFile $QOFile
done < "/home/yousen/Documents/NDLAr2x2/MuonLArSim/run_list.txt"
# edit 20250908
