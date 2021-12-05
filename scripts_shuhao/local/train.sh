DATA=/home/shuhao/Downloads/dataset
ROOT=/home/shuhao/Downloads/ObjectDetectionRL/aeroplane
mkdir -p ${ROOT}
mkdir ${ROOT}/checkpoints
mkdir ${ROOT}/stats

python main.py \
  --mode train \
  --batch_size 100 \
  --data_dir ${DATA}\
  --save_dir ${ROOT}/checkpoints/ \
  --stats_dir ${ROOT}/stats/aeroplane.json \
  --learning_rate 0.00001 \
  --epochs 2 \
  --rl_algo DuelingDQN \
  --cls aeroplane
