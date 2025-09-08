#!/bin/bash
# three task
# get the absolute directory of first argument
if [ -z "$1" ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi
if [ ! -d "$1" ]; then
    echo "Output directory $1 does not exist. Creating it."
    mkdir -p "$1"
fi
# get the absolute directory of first argument
abs_output_dir=$(realpath "$1")
# change to the directory of this script
cd "$(dirname "$0")/../plots"
root -b -q draw_totQ_totN.C
root -b -q 'draw_totQ_totN.C(true)'
mv *.png $abs_output_dir

# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250730_xoffset_1p0cm_bugfix/
# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250730_xoffset_0p5cm_bugfix/
# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250728_delay18_noreset_bugfix/
# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250728_noreset_bugfix/
# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250724_delay18_bugfix/
# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250722_bugfix/
# ./make_plots.sh ../plots/pgun_3GeV_2mm_20250805_gain_scaled_new_inputs/
