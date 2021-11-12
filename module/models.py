import torch.nn as nn
import torchvision

"""
Neural Networks we used for extracting image (based on CNN) and predicting action
These models are used by the agent. 
The code is inspired by: https://github.com/Mohammed-Deifallah/Active-Object-Localization/blob/main/utils/models.py
"""

class FeatureExtractor(nn.Module):
	def __init__(self, network='vgg16'):
		super(FeatureExtractor, self).__init__()
		if network == 'vgg16':
			model = torchvision.models.vgg16(pretrained=True)
		elif network == 'resnet50':
			model = torchvision.models.resnet50(pretrained=True)
		else:
			model = torchvision.models.alexnet(pretrained=True)
		model.eval()  # to not do dropout
		self.features = list(model.children())[0]
		self.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

	def forward(self, x):
		x = self.features(x)
		return x


class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.classifier = nn.Sequential(
			nn.Linear(in_features=81 + 25088, out_features=1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=1024, out_features=1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=1024, out_features=9)
		)

	def forward(self, x):
		return self.classifier(x)

# TODO:
class DoubleDQN(nn.Module):
	pass

class DuelingNetwork(nn.Module):
	pass