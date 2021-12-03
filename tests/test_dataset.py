import torch

from util.voc_dataset import VOCDataset, VOCClassDataset
from util import transforms as ctransforms
from util.common import plot_box, plot_tensor


def test_voc_dataset(plot=True):
    BATCH_SIZE = 1 # Must use batch size of one
    H, W = 512, 512
    l_i_transforms = [ctransforms.resize(H, W)]

    test_folder = '/home/shuhao/Downloads/dataset/test'
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


def test_voc_class_dataset(plot=True):
    BATCH_SIZE = 256 # Must use batch size of one
    H, W = 512, 512
    img_transform = ctransforms.resize_img(H, W)

    test_folder = '/home/shuhao/Downloads/dataset/test'
    test_dataset = VOCClassDataset(test_folder, img_transform=img_transform)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=4)
    print('Testing Dataloader')
    for batch_i, (images, labels) in enumerate(test_dataloader):
        print(f'Batch {batch_i}')
        print(f'Images Shape {images.shape}, Boxes Shape {labels.shape}')

    if plot:
        # Using last image and box
        img = images[0].to(torch.long)
        idx2class = VOCClassDataset.get_idx2cls()
        plot_tensor(img)
        print('Class:', idx2class[int(labels[0])])


def run_all():
    test_voc_dataset()
    test_voc_class_dataset(True)
