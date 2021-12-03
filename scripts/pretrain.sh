ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL

python ${ROOT}/main.py \
  --mode pretrain \
  -lr 1e-4 \
  --data_dir ${ROOT}/dataset \
  --save_dir ${ROOT}/checkpoints/
