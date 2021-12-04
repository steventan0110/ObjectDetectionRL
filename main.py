import os
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from module.agent import Agent
from module.pretrain import Trainer
from util import transforms as ctransforms
from util.voc_dataset import VOCDataset, VOCClassDataset


def main(args):
    H, W = args.height, args.width
    train_folder = os.path.join(args.data_dir, 'train')
    valid_folder = os.path.join(args.data_dir, 'val')
    test_folder = os.path.join(args.data_dir, 'test')

    if args.mode == "pretrain":
        img_transform = ctransforms.resize_img(H, W)
        train_dataset = VOCClassDataset(train_folder, img_transform=img_transform)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_dataset = VOCClassDataset(valid_folder, img_transform=img_transform)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_dataset = VOCClassDataset(test_folder, img_transform=img_transform)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        trainer = Trainer(train_dataloader, valid_dataloader, test_dataloader, **vars(args))
        if args.finetune:
            trainer.train()
        else:
            trainer.load_checkpoint(args.load_path_cnn)
            trainer.test()
    elif args.mode == 'train':
        transform = [ctransforms.resize(H, W)]
        cls = args.cls
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
        agent = Agent(cls, train_dataloader, valid_dataloader, **vars(args))
        print(f"Start Training for class {cls}")
        agent.train()
    elif args.mode == 'test':
        if args.load_path is None:
            raise ValueError('Need to provide parameters for test mode')

        transform = [ctransforms.resize(H, W)]
        valid_folder = os.path.join(args.data_dir, 'val')
        valid_dataset = VOCDataset(valid_folder, args.cls, label_image_transform=transform)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)
        agent = Agent(args.cls, None, valid_dataloader, **vars(args))
        print(f"Started testing class {args.cls}")
        agent.visualize()


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', '-m', choices={'train', 'test', 'pretrain'},
                        help='execution mode')
    parser.add_argument('--cls', default='aeroplane', choices=set(VOCDataset.get_classes()),
                        help='Class to use when training and testing')
    parser.add_argument('--data_dir', type=Path,
                        help='Folder where train, test, and dev data is located')
    parser.add_argument('--save_dir', type=Path,
                        help='Folder to save training and visualization data')
    parser.add_argument('--save_interval', default=5, type=int,
                        help='Intervals to save data')
    parser.add_argument('--load_path', default=None, type=Path,
                        help='Path for DQN checkpoint')
    parser.add_argument('--finetune', action='store_true',
                        help="Finetune the pretrain model")
    parser.add_argument('--load-path-cnn', default=None, type=Path,
                        help='Feature extractor checkpoint')
    parser.add_argument('--image_extractor', default='vgg16',
                        help='Feature extractor to use')
    parser.add_argument('--rl_algo', default='DQN', choices={'DQN', 'DuelingDQN'},
                        help='The reinforcement learning algorithm to use')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch when finetuning pretrained model')
    parser.add_argument('--learning_rate', '-lr', default=1e-6, type=float,
                        help='Learning rate')
    parser.add_argument('--target_update', '-tu', default=1, type=int,
                        help='Number of epochs before updating target network')
    parser.add_argument('--height', default=256, type=int,
                        help='Height to resize image to')
    parser.add_argument('--width', default=256, type=int,
                        help='Width to resize image to')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
