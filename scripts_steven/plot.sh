ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL

for class in aeroplane bicycle bird car; do
  python ${ROOT}/util/result_analyze.py \
    --mode model \
    --cls $class \
    --stats_dir ${ROOT}/stats/ \
    --save_dir ${ROOT}/stats/
done
