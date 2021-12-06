{
  CLS=$1
  MODEL=$2 # -1 for all, 0 for DQN, 1 for pretrained DQN, 2 for Dueling DQN

  # Best LR found: 0.00001
  LR=0.00001

  DATA=/raid/home/slai16/ObjectDetectionRL/dataset
  CLS_ROOT=/raid/home/slai16/ObjectDetectionRL/train_${CLS}

  if [[ $MODEL == -1 || $MODEL == 0 ]]; then
    echo "Training DQN model"
    DQN_ROOT=${CLS_ROOT}/dqn

    mkdir -p ${DQN_ROOT}
    mkdir ${DQN_ROOT}/checkpoints
    mkdir ${DQN_ROOT}/stats

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
  fi

  if [[ $MODEL == -1 || $MODEL == 1 ]]; then
    echo "Training DQN model with pretrained feature extractor"
    PRETRAINED_DQN_ROOT=${CLS_ROOT}/pretrained_dqn

    mkdir -p ${PRETRAINED_DQN_ROOT}
    mkdir ${PRETRAINED_DQN_ROOT}/checkpoints
    mkdir ${PRETRAINED_DQN_ROOT}/stats

    python main.py \
      --mode train \
      --batch_size 100 \
      --data_dir ${DATA}\
      --save_dir ${PRETRAINED_DQN_ROOT}/checkpoints/ \
      --stats_dir ${PRETRAINED_DQN_ROOT}/stats/${CLS}.json \
      --learning_rate ${LR} \
      --load-path-cnn /raid/home/slai16/ObjectDetectionRL/checkpoint_vgg_best.pt \
      --epochs 20 \
      --rl_algo DQN \
      --cls ${CLS}
  fi

  if [[ $MODEL == -1 || $MODEL == 2 ]]; then
    echo "Training Dueling DQN model"
    DUELING_DQN_ROOT=${CLS_ROOT}/dueling_dqn

    mkdir -p ${DUELING_DQN_ROOT}
    mkdir ${DUELING_DQN_ROOT}/checkpoints
    mkdir ${DUELING_DQN_ROOT}/stats

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
  fi

  exit
}