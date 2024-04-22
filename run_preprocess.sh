# !/bin/bash

rows_per_iter=10
start_point=10800
number_iters=20
for ((j=start_point; j < (start_point+rows_per_iter*number_iters); j=j+rows_per_iter))
do 
    python preprocess_pipeline.py -c $rows_per_iter -s $j
done