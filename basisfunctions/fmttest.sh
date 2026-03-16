#!/bin/bash


#set -u
# mkdir -p /scratch/p309238/archerfish/basisfunctions/

# Define parameter ranges
maxamps=( 0.01 0.1 )
minampratios=( 10 100 )

# Loop through all combinations
    for maxamp in "${maxamps[@]}"; do
        for minampratio in "${minampratios[@]}"; do


                        minamp=$(printf "%.10f" "$(echo "$maxamp / $minampratio" | bc -l)" \
         | sed 's/0*$//; s/\.$//; s/^\./0./')
         echo $minamp
    done
done

