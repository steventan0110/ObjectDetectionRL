import os
import torch
from module.models import DQN, FeatureExtractor
from module.memory import ReplayMemory
import cv2
class Agent:
	def __init__(self, train_dataloader, valid_dataloader,
	             alpha=0.2, threshold=0.6, eta=3,
	             episodes=100, max_steps_per_episode=20,
	             **kwargs):
		self.args = kwargs
		self.lr = kwargs['learning_rate']
		self.train_dataloader = train_dataloader
		self.valid_dataloader = valid_dataloader
		self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
		self.height = kwargs['height']
		self.width = kwargs['width']
		self.episodes = episodes
		self.max_steps_per_episode = max_steps_per_episode
		self.extractor = FeatureExtractor(kwargs['image_extractor']).to(self.device)
		self.extractor.eval() # do not update the pretrained CNN
		print(self.extractor)
		self.alpha = alpha
		self.threshold = threshold
		self.eta = eta
		self.epsilon = 0.2
		self.batch_size = kwargs['batch_size']
		rl_algo = kwargs['rl_algo']
		if rl_algo == 'DQN':
			self.policy_net= DQN().to(self.device)
			self.target_net = DQN() # DQN use two policy network
			self.target_net.load_state_dict(self.policy_net.state_dict())
			self.target_net.eval()
			self.target_update = 20 # TODO: hardcode for now
		print(self.policy_net)
		self.memory = ReplayMemory()
		# self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.args['learning_rate'])
		self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
			self.optimizer,
			max_lr=self.args['learning_rate'],
			epochs=self.args['max_epoch'],
			steps_per_epoch=len(self.train_dataloader))


	def get_state(self, image):
		image_feature = self.extractor(image) # (bz, 512, 16, 16) if image is not resized
		bz = image_feature.size(0)
		image_feature = image_feature.view(bz, -1)
		history = self.actions_history.view(bz, -1).type(torch.FloatTensor).to(self.device)
		state = torch.cat((image_feature, history), 1)
		return state

	def get_action(self, state):
		p = torch.rand(1).item()
		if p > self.epsilon:
			pred = self.policy_net(state)
			_, best_action = torch.max(pred, 1)
			return best_action
		else:
			return torch.randint(0, 9, (self.batch_size, ))

	def update_box(self, actions, prev_boxs, done):
		# we have a batch so each box is updated differently
		# need to take a mask that tracks done
		new_box = prev_boxs.new_full((self.batch_size, 4), 0)
		# TODO: I can't think of a way to parallelize here, since it's not computational heavy, I think use for loop
		#  here is okay
		# update action history
		self.actions_history = torch.roll(self.actions_history, -1, 1)
		for i in range(self.batch_size):
			action = int(actions[i].item())
			action_one_hot = actions.new_full((9, 1), 0).squeeze(1)
			action_one_hot[action] = 1
			self.actions_history[i][-1] = action_one_hot
			bounding_box = prev_boxs[i]
			x_min, x_max, y_min, y_max = bounding_box[0].item(), bounding_box[1].item(), bounding_box[2].item(), \
			                             bounding_box[3].item()
			x_min_out, x_max_out, y_min_out, y_max_out = x_min, x_max, y_min, y_max
			alpha_h = self.alpha * (y_max - y_min)
			alpha_w = self.alpha * (x_max - x_min)
			if action == 8:
				done[i] = True
			if action == 0:  # left
				x_min_out -= alpha_w
				x_max_out -= alpha_w
			if action == 1:  # right
				x_min_out += alpha_w
				x_max_out += alpha_w
			if action == 2:  # up
				y_min_out += alpha_h
				y_max_out += alpha_h
			if action == 3:  # down
				y_min_out -= alpha_h
				y_max_out -= alpha_h
			if action == 4: # bigger
				x_min_out -= alpha_w
				x_max_out += alpha_w
				y_min_out -= alpha_h
				y_max_out += alpha_h
			if action == 5: # smaller
				x_min_out += alpha_w
				x_max_out -= alpha_w
				y_min_out += alpha_h
				y_max_out -= alpha_h
			if action == 6: # fatter
				y_min_out += alpha_h
				y_max_out -= alpha_h
			if action == 7: # thinner
				x_min_out += alpha_w
				x_max_out -= alpha_w
			x_min_out = min(max(x_min_out,0), self.width)
			x_max_out = min(max(x_max_out, 0), self.width)
			y_min_out = min(max(y_min_out, 0), self.height)
			y_max_out = min(max(y_max_out, 0), self.height)
			new_box[i] = torch.tensor([x_min_out, x_max_out, y_min_out, y_max_out], device=self.device)

		return new_box, done

	def update_observed_region(self, image, box):
		import torchvision.transforms as T
		for i in range(self.batch_size):
			bounding_box = box[i]
			x_min, x_max, y_min, y_max = int(bounding_box[0].item()), int(bounding_box[1].item()),\
			                             int(bounding_box[2].item()), int(bounding_box[3].item())

			# resize back to the same height and width
			resize = T.Compose([
				T.Resize(self.width),
				T.CenterCrop(self.width)
			])
			# TODO: is this proper? especially if the action is fatter/thinner, the image would be stretched
			image[i] = resize(image[i, :, x_min:x_max, y_min:y_max])
		return image

	def find_closest_box(self, box, ground_truth_box):
		# TODO: implement me!
		print(box)
		print(ground_truth_box)
		raise Exception

	def compute_IOU(self, box, ground_truth):
		x_min, x_max, y_min, y_max = box[0].item(), box[1].item(), box[2].item(), box[3].item()
		x_min_gold, x_max_gold, y_min_gold, y_max_gold = ground_truth[0].item(), ground_truth[1].item(), \
		                                                 ground_truth[2].item(), ground_truth[3].item()

		box_area = (x_max - x_min) * (y_max - y_min)
		ground_truth_area = (x_max_gold - x_min_gold) * (y_max_gold - y_min_gold)
		inner_area = (min(x_max, x_max_gold) - max(x_min, x_min_gold)) * (min(y_max, y_max_gold) - max(y_min, y_min_gold))
		inner_area = max(inner_area, 0)
		union_area = box_area + ground_truth_area - inner_area
		return inner_area / union_area

	def compute_reward(self, boxes, ground_truth_boxes):
		reward = boxes.new_full((self.batch_size, 1), 0).squeeze(1)
		for i in range(self.batch_size):
			box = boxes[i]
			closet_ground_truth_box = self.find_closest_box(box, ground_truth_boxes[i])
			IOU = self.compute_IOU(box, closet_ground_truth_box)
			reward[i] = IOU
		return reward

	def train(self):
		xmin = 0.0
		xmax = self.width
		ymin = 0.0
		ymax = self.height

		for i_episode in range(self.episodes):
			for idx, (images, boxes, classes) in enumerate(self.train_dataloader):
				images, boxes, classes = images.to(self.device), boxes.to(self.device), classes.to(self.device)
				# history of past 9 actions, each consists of a one hot vec
				# TODO: this initialization might be improper for the action, should be one hot
				self.actions_history = torch.ones((self.batch_size,9,9)).to(self.device)

				prev_state = self.get_state(images) # bz x (81 + image feature size)
				original_coordinates = torch.tensor([xmin, xmax, ymin, ymax]).repeat(self.batch_size, 1).to(self.device)
				prev_box = original_coordinates
				prev_images = images

				t = 0
				done = prev_box.new_full((self.batch_size, 1), 0).bool().squeeze(1) # boolean mask
				while not torch.all(done):
					action = self.get_action(state)
					# update the bounding box and update boolean mask
					cur_box, done = self.update_box(action, prev_box, done)

					# update the observed region (change image region)
					cur_images = self.update_observed_region(prev_images, cur_box)
					next_state = self.get_state(cur_images)

					# compute reward
					reward = self.compute_reward(cur_box, boxes)
					raise Exception
					# TOOD: backprop to update policy network

					t += 1
					if t == self.max_steps_per_episode:
						break
					self.memory.push(prev_state, next_state, action, reward)
					# tracked object update for next loop
					prev_box = cur_box
					prev_images = cur_images
					prev_state = next_state



			if i_episode % self.target_update == 0:
				self.target_net.load_state_dict(self.policy_net.state_dict())


	def save_checkpoint(self, epoch, loss, val_loss, dir):
		save_path = os.path.join(dir, 'checkpoint' + str(epoch) + '.pt')
		torch.save({
			'epoch': epoch,
			'loss': loss,
			'val_loss': val_loss,
			'policy_network': self.policy_net.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
		}, save_path)

	def load_checkpoint(self, dir):
		checkpoint = torch.load(dir, map_location=self.device)
		self.policy_net.load_state_dict(checkpoint['policy_network'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


