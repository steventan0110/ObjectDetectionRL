from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from module.agent import Agent
from util import transforms as ctransforms
from util.voc_dataset import VOCDataset


def main(args):
    print(args)
    H, W = args.height, args.width
    transform = [ctransforms.resize(H, W)]

    if args.mode == 'train':
        # begin train loop
        train_folder = f'{args.data_dir}/train'
        train_dataset = VOCDataset(train_folder, label_image_transform=transform)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=4)
        valid_folder = f'{args.data_dir}/val'
        valid_dataset = VOCDataset(valid_folder, label_image_transform=transform)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=4)

        # agent is the trainer that control training for 1 epoch
        agent = Agent(train_dataloader, valid_dataloader, **vars(args))

        # TODO: enable save and load for trainer
        # if args.load_dir:
        #     trainer.load_checkpoint(args.load_dir)
        agent.train()

        # TODO: uncomment after train and validate function is fully working
        # for i in range(args.max_epoch):
        #     # trainer only responsible for training one epoch
        #     loss, val_loss = trainer.train()
        #     print('Epoch{}, loss:{}, validation loss{}'.format(i + 1, loss, val_loss), flush=True)
        #     if i % args.save_interval == 0:
        #         trainer.save_checkpoint(i, loss, val_loss, args.save_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', '-m', choices={'train', 'test', 'interactive'}, help="execution mode")
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--load-dir', default=None, type=Path)
    parser.add_argument('--cpu', default=False, action="store_true")
    parser.add_argument('--image-extractor', default="vgg16")
    parser.add_argument('--rl-algo', default="DQN")
    parser.add_argument('--max-epoch', default=100, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
    parser.add_argument('--save-interval', default=1, type=int)
    parser.add_argument('--beam-size', default=5, type=int)
    parser.add_argument('--height', default=256, type=int)
    parser.add_argument('--width', default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)