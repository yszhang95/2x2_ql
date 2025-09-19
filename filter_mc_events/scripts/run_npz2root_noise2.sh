#!/bin/bash

while IFS= read -r InFile; do
    echo "Processing $InFile"
    SrcFile="$InFile"
    bname=$(basename "$InFile")
    odir="${bname%.hdf5}"

    SrcBase=$(basename "$SrcFile")
    HitFileDir="${SrcBase%.hdf5}"
    HitFile="/home/yousen/Documents/NDLAr2x2/tred/tests/playground/for_batch_pgun/${HitFileDir}/pgun_mu_3GeV_2mm_noise2.npz"
    HODir=$(dirname ${HitFile})
    HOFile="${HODir}/pgun_mu_3GeV_2mm_noise2_hits.hdf5"

    if [[ ! -d $HitFileDir ]]; then
        echo "Output directory $HitFileDir does not exist. Creating it."
        echo $HitFileDir
        mkdir "$HitFileDir"
    fi

    if [[ ! -f $HitFile ]]; then
        echo "Hits hdf5 file $HitFile does not exists. Skipping."
        continue
    fi

    echo "Converting npz to hdf5 for $HitFile"
    # last argument is 1 to fix mismatching tpc
    uv run ../many_muons.py $HitFile $HODir  "hits" 1
    uv run ../many_muons.py $HitFile $HODir  "effq" 1

    # if [[ ! -f $HOFile ]]; then
    #     echo "Selected hits hdf5 file $HOFile does not exists. Skipping."
    #     continue
    # fi

    QOFile=${HOFile/_hits.hdf5/_effq.hdf5}
    QODir=$(dirname ${QOFile})
    HOFileSelected="${HODir}/pgun_mu_3GeV_2mm_noise2_20250916_selected_hits.hdf5"
    QOFileSelected=${HOFileSelected/_hits/_effq}

    echo "Selecting data in hdf5"
    # echo "Hits"
    uv run ../pick_pgun2.py --dtype=hits $SrcFile $HOFile $HOFileSelected
    # echo "EffQ"
    uv run ../pick_pgun2.py --dtype=effq $SrcFile $QOFile $QOFileSelected

    echo "Converting data to root"
    uv run ../filter_data.py $HOFileSelected ${HODir}/pgun_mu_3GeV_2mm_noise2_20250916_selected_hits.root
    uv run ../filter_data.py $QOFileSelected ${QODir}/pgun_mu_3GeV_2mm_noise2_20250916_selected_effq.root
done < "/home/yousen/Documents/NDLAr2x2/MuonLArSim/run_list.txt"
# edit 20250916
