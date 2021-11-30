ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL

python ${ROOT}/main.py \
  --mode train \
  --data_dir ${ROOT}/dataset \
  --save_dir ${ROOT}checkpoints/
