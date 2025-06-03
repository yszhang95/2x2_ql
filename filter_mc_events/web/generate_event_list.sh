#!/bin/bash
$(ls *evt*hdf5 > event_list.txt)
perl -i -pe 's/many_muon_hits_evt(\d+)\.hdf5/$1/g' event_list.txt
