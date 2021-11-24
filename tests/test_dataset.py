import torch

from util.voc_dataset import VOCDataset
from util import transforms as ctransforms
from tests.test_transforms import plot_box


def test_voc_dataset(plot=True):
    BATCH_SIZE = 1 # Must use batch size of one
    H, W = 512, 512
    l_i_transforms = [ctransforms.resize(H, W)]

    test_folder = '/home/jun/Downloads/dataset/test/'
    cls = 'aeroplane'
    test_dataset = VOCDataset(test_folder, cls, label_image_transform=l_i_transforms)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=4)
    print_every = 100
    print('Testing Dataloader')
    for batch_i, (images, boxes) in enumerate(test_dataloader):
        if batch_i % print_every == 0:
            print(f'Batch {batch_i}')
            print(f'Images Shape {images.shape}, Boxes Shape {boxes.shape}')

    if plot:
        # Using last image and box
        box = boxes[0][0]
        img = images[0].to(torch.long)
        plot_box(img, box)


def run_all():
    test_voc_dataset()
