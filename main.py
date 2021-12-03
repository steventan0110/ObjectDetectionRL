import os
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from module.agent import Agent
from module.models import FeatureExtractor
from util import transforms as ctransforms
from util.voc_dataset import VOCDataset
from tqdm import tqdm

def main(args):
    H, W = args.height, args.width
    transform = [ctransforms.resize(H, W)]

    if args.mode == "pretrain":
        train_folder = os.path.join(args.data_dir, 'train')
        valid_folder = os.path.join(args.data_dir, 'val')
        classes = VOCDataset.get_classes()
        vgg = FeatureExtractor(freeze=False)
        for cls in classes:
            train_dataset = VOCDataset(train_folder, cls, label_image_transform=transform)
            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4)
            for image, boxes in tqdm(train_dataloader):
                out = vgg(image)
                print(out.shape)
                raise Exception
            print(cls)

    if args.mode == 'train':
        train_folder = os.path.join(args.data_dir, 'train')
        valid_folder = os.path.join(args.data_dir, 'val')
        # classes = VOCDataset.get_classes()
        # for cls in classes:
        for cls in ['aeroplane']:
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
    elif args.mode == 'test':
        if args.load_path is None:
            raise ValueError('Need to provide parameters for test mode')

        cls = 'aeroplane' # TODO do not hardcode in the future
        valid_folder = os.path.join(args.data_dir, 'val')
        valid_dataset = VOCDataset(valid_folder, cls, label_image_transform=transform)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)
        agent = Agent(cls, None, valid_dataloader, **vars(args))
        print(f"Start Training for class {cls}")
        agent.visualize()



def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', '-m', choices={'train', 'test', 'interactive', 'pretrain'}, help='execution mode')
    parser.add_argument('--data_dir', type=Path, help='Folder where train, test, and dev data is located')
    parser.add_argument('--save_dir', type=Path, help='Folder to save training and visualization data')
    parser.add_argument('--save_interval', default=5, type=int, help='Intervals to save data')
    parser.add_argument('--load_path', default=None, type=Path, help='Folder where checkpoint params are located')
    parser.add_argument('--image_extractor', default='vgg16', help='Feature extractor to use')
    parser.add_argument('--rl_algo', default='DQN', help='The reinforcement learning algorithm to use')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--target_update', '-tu', default=1, type=int, help='Number of epochs before updating target '
                                                                            'network')
    parser.add_argument('--height', default=256, type=int, help='Height to resize image to')
    parser.add_argument('--width', default=256, type=int, help='Width to resize image to')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)