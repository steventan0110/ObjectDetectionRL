import numpy as np
import torchvision
import json
import torch
import os

from torch.utils.data import Dataset


'''
View test_dataset for example usage
'''


class VOCDataset(Dataset):
	def __init__(self, data_folder, cls, label_image_transform=None, img_transform=None):
		# Be careful with transformations that change location of bounding box!

		# For transformations that change label locations (i.e. rotation)
		self.label_image_transform = label_image_transform
		# For transformations that do not change label location (i.e. blur)
		self.img_transform = img_transform

		assert cls in self.get_classes()
		self.cls = cls
		self.labels_dict, self.filenames = self.parse_labels(data_folder)
		self.image_folder = os.path.join(data_folder, 'images')

	@staticmethod
	def get_class2idx():
		cls2idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2,
				   'boat': 3, 'bottle': 4, 'bus': 5,
				   'car': 6, 'cat': 7, 'chair': 8,
				   'cow': 9, 'diningtable': 10, 'dog': 11,
				   'horse': 12, 'motorbike': 13, 'person': 14,
				   'pottedplant': 15, 'sheep': 16, 'sofa': 17,
				   'train': 18, 'tvmonitor': 19}
		return cls2idx

	@staticmethod
	def get_idx2cls():
		cls2idx = VOCDataset.get_class2idx()
		reverse = [(idx, cls) for cls, idx in cls2idx.items()]
		idx2cls = dict(reverse)
		return idx2cls

	@staticmethod
	def get_classes():
		cls2idx = VOCDataset.get_class2idx()
		classes = list(cls2idx.keys())
		classes.sort()
		return classes

	def parse_labels(self, data_path):
		json_path = os.path.join(data_path, 'sorted_labels.json')
		with open(json_path) as f:
			labels = json.load(f)
			cls_labels_dict = labels[self.cls]
			filenames = list(cls_labels_dict.keys())
			return cls_labels_dict, filenames

	def __getitem__(self, idx):
		img_name = self.filenames[idx]

		# Formatting label information
		boxes = self.labels_dict[img_name]
		boxes = torch.tensor(boxes)

		# Formatting image information
		img_path = os.path.join(self.image_folder, img_name)
		img = torchvision.io.read_image(img_path)

		if self.label_image_transform:
			for transform in self.label_image_transform:
				img, boxes = transform(img, boxes)
		if self.img_transform:
			img = self.img_transform(img)

		img = img.to(torch.float32)
		boxes = torch.round(boxes).to(torch.long) if boxes.dtype != torch.long else boxes

		return img, boxes

	def __len__(self):
		return len(self.filenames)


class VOCClassDataset(Dataset):
	def __init__(self, data_folder, img_transform=None):
		self.img_transform = img_transform
		self.data = self.parse_labels(data_folder)

	@staticmethod
	def get_class2idx():
		return VOCDataset.get_class2idx()

	@staticmethod
	def get_classes():
		return VOCDataset.get_classes()

	@staticmethod
	def get_idx2cls():
		return VOCDataset.get_idx2cls()

	def parse_labels(self, data_path):
		json_path = os.path.join(data_path, 'labels.json')
		with open(json_path) as f:
			file2objs = json.load(f)
			data = []
			cls2id = self.get_class2idx()
			num_classes = len(cls2id)
			# Creating distribution of data
			for filename, lis_of_objs in file2objs.items():
				counts = np.zeros(num_classes)
				for obj in lis_of_objs:
					cls, pose, box = obj['name'], obj['pose'], obj['box']
					counts[cls2id[cls]] += 1

				entry = {
					'distribution': counts / len(lis_of_objs),
					'file_path': os.path.join(data_path, 'images', filename)
				}
				data.append(entry)
			return data

	def __getitem__(self, idx):
		entry = self.data[idx]
		distribution = entry['distribution']
		img_path = entry['file_path']

		img = torchvision.io.read_image(img_path)
		cls_indices = np.arange(len(distribution))
		cls = np.random.choice(cls_indices, 1, p=distribution)

		if self.img_transform:
			img = self.img_transform(img)

		img = img.to(torch.float32)
		cls = torch.tensor(cls).type(torch.LongTensor)
		return img, cls

	def __len__(self):
		return len(self.data)