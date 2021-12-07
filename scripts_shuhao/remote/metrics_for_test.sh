{
  CLS=$1
  MODEL=$2 # 0=dqn, 1=pretrained_model, 2=dueling_dqn

  ROOT=/raid/home/slai16/ObjectDetectionRL
  DATA=${ROOT}/dataset

  if [[ $MODEL == 0 ]]; then
    echo "Getting test metrics for DQN model"
    export CUDA_VISIBLE_DEVICES=0
    echo "Current using CUDA device $CUDA_VISIBLE_DEVICES"
    python main.py \
      --cls ${CLS} \
      --mode test \
      --data_dir ${DATA} \
      --rl_algo DQN \
      --load_path ${ROOT}/validation_lr_0_00001/train_${CLS}_best_lr/dqn/checkpoints/checkpoint_${CLS}_best.pt
  fi

  if [[ $MODEL == 1 ]]; then
    echo "Getting test metrics for pretrained DQN model"
    export CUDA_VISIBLE_DEVICES=1
    echo "Current using CUDA device $CUDA_VISIBLE_DEVICES"
    python main.py \
      --cls ${CLS} \
      --mode test \
      --data_dir ${DATA} \
      --rl_algo DQN \
      --load-path-cnn /raid/home/slai16/ObjectDetectionRL/checkpoint_vgg_best.pt \
      --load_path ${ROOT}/validation_lr_0_00001/train_${CLS}_best_lr/pretrained_dqn/checkpoints/checkpoint_${CLS}_best.pt
  fi

  if [[ $MODEL == 2 ]]; then
    echo "Getting test metrics for Dueling DQN model"
    export CUDA_VISIBLE_DEVICES=2
    echo "Current using CUDA device $CUDA_VISIBLE_DEVICES"
    python main.py \
      --cls ${CLS} \
      --mode test \
      --data_dir ${DATA} \
      --rl_algo DuelingDQN \
      --load_path ${ROOT}/validation_lr_0_00001/train_${CLS}_best_lr/dueling_dqn/checkpoints/checkpoint_${CLS}_best.pt
  fi

  exit
}