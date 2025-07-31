#!/bin/bash
# three task
root -b -q draw_totQ_totN.C
root -b -q 'draw_totQ_totN.C(true)'
mv *.png $1
