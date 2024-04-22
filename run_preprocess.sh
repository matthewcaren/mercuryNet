# !/bin/bash
start=`date +%s`
rows_per_iter=10
start_point=11200
number_iters=10
for ((j=start_point; j < (start_point+rows_per_iter*number_iters); j=j+rows_per_iter))
do 
    python preprocess_pipeline.py -c $rows_per_iter -s $j
done
end=`date +%s`
runtime=$((end-start))
echo $runtime