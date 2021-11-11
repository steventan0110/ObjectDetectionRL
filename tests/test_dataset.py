import torch

from util.voc_dataset import VOCDataset
from util import transforms as ctransforms


def test_voc_dataset():
    BATCH_SIZE = 32
    H, W = 512, 512
    l_i_transforms = [ctransforms.resize(H, W)]

    test_folder = '/home/shuhao/Downloads/dataset/test'
    test_dataset = VOCDataset(test_folder, label_image_transform=l_i_transforms)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=4)

    print_every = 100
    print('Starting Testing Dataloader')
    for batch_i, (images, boxes, classes) in enumerate(test_dataloader):
        if batch_i % print_every == 0:
            print(f'Batch {batch_i}')
            print(f'Images Shape {images.shape}, Boxes Shape {boxes.shape}, Classes shape {classes.shape}')


def run_all():
    test_voc_dataset()
