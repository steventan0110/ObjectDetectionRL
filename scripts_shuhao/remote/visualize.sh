CLS=$1
MODEL=$2 # 0=dqn, 1=pretrained_model, 2=dueling_dqn

ROOT=/raid/home/slai16/ObjectDetectionRL
DATA=${ROOT}/dataset
OUTPUT=${ROOT}/visuals

if [[ $MODEL == 0 ]]; then
  echo "Visualizing DQN model"
  python main.py \
    --cls ${CLS} \
    --mode visualize \
    --data_dir ${DATA} \
    --rl_algo DQN \
    --load_path ${ROOT}/validation_lr_0_00001/train_${CLS}_best_lr/dqn/checkpoints/checkpoint_${CLS}_best.pt \
    --save_dir ${OUTPUT}/${CLS}/dqn
fi

if [[ $MODEL == 1 ]]; then
  echo "Visualizing pretrained DQN model"
  python main.py \
    --cls ${CLS} \
    --mode visualize \
    --data_dir ${DATA} \
    --rl_algo DQN \
    --load-path-cnn /raid/home/slai16/ObjectDetectionRL/checkpoint_vgg_best.pt \
    --load_path ${ROOT}/validation_lr_0_00001/train_${CLS}_best_lr/pretrained_dqn/checkpoints/checkpoint_${CLS}_best.pt \
    --save_dir ${OUTPUT}/${CLS}/pretrained_dqn
fi

if [[ $MODEL == 2 ]]; then
  echo "Visualizing Dueling DQN model"
  python main.py \
    --cls ${CLS} \
    --mode visualize \
    --data_dir ${DATA} \
    --rl_algo DuelingDQN \
    --load_path ${ROOT}/validation_lr_0_00001/train_${CLS}_best_lr/dueling_dqn/checkpoints/checkpoint_${CLS}_best.pt \
    --save_dir ${OUTPUT}/${CLS}/dueling_dqn
fi