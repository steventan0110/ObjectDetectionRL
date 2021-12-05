LR=$1

DATA=/raid/home/slai16/ObjectDetectionRL/dataset
ROOT=/raid/home/slai16/ObjectDetectionRL/aeroplane_lr_${LR}
mkdir ${ROOT}
mkdir ${ROOT}/checkpoints
mkdir ${ROOT}/stats

python main.py \
  --mode train \
  --batch_size 100 \
  --data_dir ${DATA}\
  --save_dir ${ROOT}/checkpoints/ \
  --stats_dir ${ROOT}/stats/aeroplane.json \
  --learning_rate ${LR} \
  --epochs 25 \
  --rl_algo DQN \
  --cls aeroplane
