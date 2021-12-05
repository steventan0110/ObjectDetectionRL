import os
from pathlib import Path
import argparse
from voc_dataset import VOCDataset, VOCClassDataset
import json
import matplotlib.pyplot as plt


def get_threshold_precision(inp):
    arr1, arr2, arr3, arr4, arr5 = [], [], [], [], []
    for i in range(len(inp)):
        arr1.append(inp[i][0])
        arr2.append(inp[i][1])
        arr3.append(inp[i][2])
        arr4.append(inp[i][3])
        arr5.append(inp[i][4])
    return arr1, arr2, arr3, arr4, arr5

def get_lr_threshold_precision(inp):
    from collections import defaultdict
    out = dict()
    for lr in ['1e-5', '1e-6', '1e-7']:
        out[lr] = [[] for _ in range(5)]

    for lr in ['1e-5', '1e-6', '1e-7']:
        precision_arr = inp[lr]['precision']
        for i in range(len(precision_arr)):
            for j in range(5):
                out[lr][j].append(precision_arr[i][j])
    return out

def main(args):
    dir = args.stats_dir
    if args.mode == 'model':
        models = ['dqn', 'dueling_dqn', 'pretrained_dqn']
        # plot three models's validation result
        cls_folder = f'{dir}/train_{args.cls}_best_lr'
        with open(f'{cls_folder}/dqn/stats/{args.cls}.json', 'r') as f1:
            dqn_json = json.load(f1)
        with open(f'{cls_folder}/dueling_dqn/stats/{args.cls}.json', 'r') as f2:
            dueling_json = json.load(f2)
        # with open(f'{cls_folder}/pretrained_dqn/stats/{args.cls}.json', 'r') as f3:
        #     pretrain_json = json.load(f3)

        save_dir = f'{args.save_dir}/model_compare/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # ---------------------------------- Train LOSS ------------------------------------------------#
        plt.figure()
        plt.plot(dqn_json['train_loss'], label='DQN Train Loss')
        plt.plot(dueling_json['train_loss'], label='Dueling DQN Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_dir}/train_loss.png')
        # ---------------------------------- Precision ------------------------------------------------#

        dqn1, dqn2, dqn3, dqn4, dqn5 = get_threshold_precision(dqn_json['precision'])
        ddqn1, ddqn2, ddqn3, ddqn4, ddqn5 = get_threshold_precision(dueling_json['precision'])
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(16, 5))
        ax1.plot(dqn1, label='DQN (t=0.1)')
        ax1.plot(ddqn1, label='Dueling DQN (threshold=0.1)')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Precision')
        ax1.legend(loc="upper right")

        ax2.plot(dqn2, label='DQN (t=0.2)')
        ax2.plot(ddqn2, label='Dueling DQN (threshold=0.2)')
        ax2.set_xlabel('Epochs')
        ax2.legend(loc="upper right")

        ax3.plot(dqn3, label='DQN (t=0.3)')
        ax3.plot(ddqn3, label='Dueling DQN (threshold=0.3)')
        ax3.set_xlabel('Epochs')
        ax3.legend(loc="upper right")

        ax4.plot(dqn4, label='DQN (t=0.4)')
        ax4.plot(ddqn4, label='Dueling DQN (threshold=0.4)')
        ax4.set_xlabel('Epochs')
        ax4.legend(loc="upper right")

        ax5.plot(dqn5, label='DQN (t=0.5)')
        ax5.plot(ddqn5, label='Dueling DQN (threshold=0.5)')
        ax5.set_xlabel('Epochs')
        ax5.legend(loc="upper right")
        #plt.show()
        plt.savefig(f'{save_dir}/precision.png')

        # ---------------------------------- Reward ------------------------------------------------#
        plt.figure()
        plt.plot(dqn_json['reward'], label='DQN Reward')
        plt.plot(dueling_json['reward'], label='Dueling DQN Reward')
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_dir}/reward.png')

        # ---------------------------------- IOU ------------------------------------------------#
        plt.figure()
        plt.plot(dqn_json['IOU'], label='DQN IOU')
        plt.plot(dueling_json['IOU'], label='Dueling DQN IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_dir}/IOU.png')
    else:
        # plot the curve for three learning rate
        # no data for it, ignore for now
        result = dict()
        for lr in ['1e-5', '1e-6', '1e-7']:
            file = f'{dir}/{args.cls}_{lr}/{args.cls}.json'
            with open(file, 'r') as f:
                temp = json.load(f)
                result[lr] = temp

        save_dir = f'{args.save_dir}/lr_compare/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # ---------------------------------- Train LOSS ------------------------------------------------#
        plt.figure()
        for lr in ['1e-5', '1e-6', '1e-7']:
            plt.plot(result[lr]['train_loss'], label=f'Train Loss (lr = {lr})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_dir}/train_loss.png')

        # ---------------------------------- Precision ------------------------------------------------#

        data = get_lr_threshold_precision(result)
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 5))
        for lr in ['1e-5', '1e-6', '1e-7']:
            ax1.plot(data[lr][0], label=f'lr={lr}, t=0.1')
            ax2.plot(data[lr][1], label=f'lr={lr}, t=0.2')
            ax3.plot(data[lr][2], label=f'lr={lr}, t=0.3')
            ax4.plot(data[lr][3], label=f'lr={lr}, t=0.4')
            ax5.plot(data[lr][4], label=f'lr={lr}, t=0.5')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Precision')
        ax1.legend(loc="upper right")
        ax2.set_xlabel('Epochs')
        ax2.legend(loc="upper right")
        ax3.set_xlabel('Epochs')
        ax3.legend(loc="upper right")
        ax4.set_xlabel('Epochs')
        ax4.legend(loc="upper right")
        ax5.set_xlabel('Epochs')
        ax5.legend(loc="upper right")
        # plt.show()

        plt.savefig(f'{save_dir}/precision.png')

        # ---------------------------------- Reward ------------------------------------------------#
        plt.figure()
        for lr in ['1e-5', '1e-6', '1e-7']:
            plt.plot(result[lr]['reward'], label=f'Reward (lr = {lr})')
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_dir}/reward.png')

        # ---------------------------------- IOU ------------------------------------------------#
        plt.figure()
        for lr in ['1e-5', '1e-6', '1e-7']:
            plt.plot(result[lr]['IOU'], label=f'IOU (lr = {lr})')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_dir}/IOU.png')



def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    # lr: compare across learning rate, model: compare across three model
    parser.add_argument('--mode', '-m', choices={'lr', 'model'},
                        help='execution mode')
    parser.add_argument('--cls', default='aeroplane', choices=set(VOCDataset.get_classes()),
                        help='Class to use when training and testing')
    parser.add_argument('--stats_dir', type=Path, default=None,
                        help='Folder to save stats about training and validation')
    parser.add_argument('--save_dir', type=Path, default=None,
                        help='Folder to save output image')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)