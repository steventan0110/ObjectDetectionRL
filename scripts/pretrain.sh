ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL

python ${ROOT}/main.py \
  --mode pretrain \
  --data_dir ${ROOT}/dataset \
  --save_dir ${ROOT}/checkpoints/
