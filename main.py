import os
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from module.agent import Agent
from util import transforms as ctransforms
from util.voc_dataset import VOCDataset


def main(args):
    H, W = args.height, args.width
    transform = [ctransforms.resize(H, W)]

    if args.mode == 'train':
        train_folder = os.path.join(args.data_dir, 'train')
        valid_folder = os.path.join(args.data_dir, 'val')
        cls2idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2,
                   'boat': 3, 'bottle': 4, 'bus': 5,
                   'car': 6, 'cat': 7, 'chair': 8,
                   'cow': 9, 'diningtable': 10, 'dog': 11,
                   'horse': 12, 'motorbike': 13, 'person': 14,
                   'pottedplant': 15, 'sheep': 16, 'sofa': 17,
                   'train': 18, 'tvmonitor': 19}
        for cls in cls2idx.keys():
            train_dataset = VOCDataset(train_folder, cls, label_image_transform=transform)
            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4)

            valid_dataset = VOCDataset(valid_folder, cls, label_image_transform=transform)
            valid_dataloader = DataLoader(dataset=valid_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4)
            # agent is the trainer that control training for 1 epoch
            agent = Agent(cls, train_dataloader, valid_dataloader, **vars(args))
            print(f"Start Training for class {cls}")
            agent.train()
            print()


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', '-m', choices={'train', 'test', 'interactive'}, help='execution mode')
    parser.add_argument('--data_dir', type=Path, help='Folder where train, test, and dev data is located')
    parser.add_argument('--save_dir', type=Path, help='Folder to save training data')
    parser.add_argument('--save_interval', default=5, type=int, help='Intervals to save data')
    parser.add_argument('--load_dir', default=None, type=Path, help='Folder where checkpoint params are located')
    parser.add_argument('--image_extractor', default='vgg16', help='Feature extractor to use')
    parser.add_argument('--rl_algo', default='DQN', help='The reinforcement learning algorithm to use')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--target_update', '-tu', default=20, type=int, help='Number of epochs before updating target network')
    parser.add_argument('--height', default=256, type=int, help='Height to resize image to')
    parser.add_argument('--width', default=256, type=int, help='Width to resize image to')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)