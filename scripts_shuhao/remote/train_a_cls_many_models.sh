# Best LR found: 0.00001

# Trying aeroplane, person, car, chair
CLS=$1
LR=0.000001

DATA=/raid/home/slai16/ObjectDetectionRL/dataset
CLS_ROOT=/raid/home/slai16/ObjectDetectionRL/train_${CLS}_best_lr

DQN_ROOT=${CLS_ROOT}/dqn
PRETRAINED_DQN_ROOT=${CLS_ROOT}/pretrained_dqn
DUELING_DQN_ROOT=${CLS_ROOT}/dueling_dqn

mkdir -p ${DQN_ROOT}
mkdir ${DQN_ROOT}/checkpoints
mkdir ${DQN_ROOT}/stats

mkdir -p ${PRETRAINED_DQN_ROOT}
mkdir ${PRETRAINED_DQN_ROOT}/checkpoints
mkdir ${PRETRAINED_DQN_ROOT}/stats

mkdir -p ${DUELING_DQN_ROOT}
mkdir ${DUELING_DQN_ROOT}/checkpoints
mkdir ${DUELING_DQN_ROOT}/stats

python main.py \
  --mode train \
  --batch_size 100 \
  --data_dir ${DATA}\
  --save_dir ${DQN_ROOT}/checkpoints/ \
  --stats_dir ${DQN_ROOT}/stats/${CLS}.json \
  --learning_rate ${LR} \
  --epochs 20 \
  --rl_algo DQN \
  --cls ${CLS}

python main.py \
  --mode train \
  --batch_size 100 \
  --data_dir ${DATA}\
  --save_dir ${PRETRAINED_DQN_ROOT}/checkpoints/ \
  --stats_dir ${PRETRAINED_DQN_ROOT}/stats/${CLS}.json \
  --learning_rate ${LR} \
  --load-path-cnn /raid/home/slai16/ObjectDetectionRL/checkpoint_vgg_best.pt \
  --epochs 25 \
  --rl_algo DQN \
  --cls ${CLS}

python main.py \
  --mode train \
  --batch_size 100 \
  --data_dir ${DATA}\
  --save_dir ${DUELING_DQN_ROOT}/checkpoints/ \
  --stats_dir ${DUELING_DQN_ROOT}/stats/${CLS}.json \
  --learning_rate ${LR} \
  --epochs 20 \
  --rl_algo DuelingDQN \
  --cls ${CLS}
