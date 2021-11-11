# Load datasets
import torch
from torch.utils.data import Dataset
from skimage import io
import os
import json
import numpy as np
from torchvision import transforms
import pickle


class VOLDataset(Dataset):
	def __init__(self, input_dir, op, transform=None):
		self.op = op
		try:
			if self.op == 'train':
				self.data_dir = os.path.join(input_dir, 'train')
			elif self.op == 'valid':
				self.data_dir = os.path.join(input_dir, 'valid')
			elif self.op == 'test':
				self.data_dir = os.path.join(input_dir, 'test')
			else:
				raise ValueError
		except ValueError:
			print('op should be either train, val or test!')

		if not transform:
			self.transform = transforms.Compose([transforms.ToTensor()])
		else:
			self.transform = transform

		with open(f'{self.data_dir}/labels.json') as f:
			self.label = json.load(f)

	def __getitem__(self, index):
		img = io.imread(os.path.join(self.data_dir, 'images', f'{index}.jpg'))
		out_img = self.transform(img)
		label_list = self.label[f'{index}.jpg']
		return out_img, label_list

	def __len__(self):
		length = 0
		for base, dirs, files in os.walk(f'{self.data_dir}/images'):
			for _ in files:
				length += 1
		return length


# need collate function to pad the sentences
def collate_fn(data):
	image, bounding_box = zip(*data)
	# TODO: maybe we don't need collate, just put this function here for future use
	pass


def get_loader(input_dir, op, transform, batch_size, shuffle=False):
	dataset = VOLDataset(input_dir=input_dir, op=op, transform=transform)
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
	                                          batch_size=batch_size,
	                                          shuffle=shuffle,
	                                          collate_fn=collate_fn)
	return data_loader


if __name__ == '__main__':
	# test data loader
	input_dir = '/home/steven/Code/GITHUB/ObjectDetectionRL/dataset'
	op = 'test'
	batch_size = 2
	resize = 256
	crop_size = 224
	# transform = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	# resize is necessary!
	# 	transforms.Resize(resize),
	# 	transforms.RandomCrop(crop_size),
	# 	transforms.RandomHorizontalFlip(),
	# 	transforms.Normalize((0.485, 0.456, 0.406),
	# 	                     (0.229, 0.224, 0.225))])


	dataloader = get_loader(input_dir, op, None, batch_size)

	for item in dataloader:
		img, caption = item
		print(img.shape)
		print(caption)
		break
