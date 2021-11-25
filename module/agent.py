import os
import torch
from module.models import DQN, FeatureExtractor
from module.memory import ReplayMemory
import cv2
from torch.autograd import Variable


class Agent:

    def __init__(self, train_dataloader, valid_dataloader,
                 alpha=0.2, threshold=0.6, eta=3,
                 episodes=100, max_steps_per_episode=20,
                 **kwargs):

        self.args = kwargs
        self.gamma = 0.7
        self.lr = kwargs['learning_rate']
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.extractor = FeatureExtractor(kwargs['image_extractor']).to(self.device)
        self.extractor.eval()  # do not update the pretrained CNN
        print(self.extractor)
        self.alpha = alpha
        self.threshold = threshold
        self.eta = eta
        self.epsilon = 0.2
        self.batch_size = kwargs['batch_size']
        rl_algo = kwargs['rl_algo']
        if rl_algo == 'DQN':
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)  # DQN use two policy network
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.target_update = 20  # TODO: hardcode for now
        print(self.policy_net)
        self.memory = ReplayMemory(1000) # capacity = 1000
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.args['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args['learning_rate'],
            epochs=self.args['max_epoch'],
            steps_per_epoch=len(self.train_dataloader))

    def get_state(self, image):
        image_feature = self.extractor(image)  # (bz, 512, 16, 16) if image is not resized
        image_feature = image_feature.view(self.batch_size, -1)
        history = self.actions_history.view(self.batch_size, -1).type(torch.FloatTensor).to(self.device)
        state = torch.cat((image_feature, history), 1)
        return state

    def get_action(self, state):
        p = torch.rand(1).item()
        if p > self.epsilon:
            pred = self.policy_net(state)
            _, best_action = torch.max(pred, 1)
            return best_action
        else:
            return torch.randint(0, 9, (self.batch_size,))

    def update_box(self, action, prev_box):
        # TODO: I can't think of a way to parallelize here, since it's not computational heavy, I think use for loop
        #  here is okay
        # update action history, roll the history so that the earliest action is no longer tracked
        self.actions_history = torch.roll(self.actions_history, 1, 0)
        action_one_hot = self.actions_history.new_full((9, 1), 0).squeeze(1)
        action_one_hot[action] = 1
        self.actions_history[0] = action_one_hot
        # print(action, action_one_hot, self.actions_history)

        bounding_box = prev_box
        x_min, x_max, y_min, y_max = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x_min_out, x_max_out, y_min_out, y_max_out = x_min, x_max, y_min, y_max
        alpha_h = self.alpha * (y_max - y_min)
        alpha_w = self.alpha * (x_max - x_min)
        if action == 8:
            return None, True
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
        if action == 4:  # bigger
            x_min_out -= alpha_w
            x_max_out += alpha_w
            y_min_out -= alpha_h
            y_max_out += alpha_h
        if action == 5:  # smaller
            x_min_out += alpha_w
            x_max_out -= alpha_w
            y_min_out += alpha_h
            y_max_out -= alpha_h
        if action == 6:  # fatter
            y_min_out += alpha_h
            y_max_out -= alpha_h
        if action == 7:  # thinner
            x_min_out += alpha_w
            x_max_out -= alpha_w
        x_min_out = min(max(x_min_out, 0), self.width)
        x_max_out = min(max(x_max_out, 0), self.width)
        y_min_out = min(max(y_min_out, 0), self.height)
        y_max_out = min(max(y_max_out, 0), self.height)
        new_box = [x_min_out, x_max_out, y_min_out, y_max_out]
        return new_box, False

    def update_observed_region(self, image, box):
        import torchvision.transforms as T
        x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # resize back to the same height and width
        resize = T.Compose([
            T.Resize(self.width),
            T.CenterCrop(self.width)
        ])
        # TODO: is this proper? especially if the action is fatter/thinner, the image would be stretched
        # sometimes get division by 0 error
        image = resize(image[:, :, x_min:x_max, y_min:y_max])
        return image

    def find_closest_box(self, box, ground_truth_box):
        num_box = ground_truth_box.size(0)
        max_area = -float('inf')
        best_box = None
        for i in range(num_box):
            true_box = [ground_truth_box[i][0].item(), ground_truth_box[i][1].item(), ground_truth_box[i][2].item(),
                        ground_truth_box[i][3].item()]
            area = self.compute_IOU(box, true_box)
            if area > max_area:
                max_area = area
                best_box = true_box
        return max_area, best_box

    def compute_IOU(self, box, ground_truth):
        x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
        x_min_gold, x_max_gold, y_min_gold, y_max_gold = ground_truth[0], ground_truth[1], \
                                                         ground_truth[2], ground_truth[3]

        box_area = (x_max - x_min) * (y_max - y_min)
        ground_truth_area = (x_max_gold - x_min_gold) * (y_max_gold - y_min_gold)
        inner_area = (min(x_max, x_max_gold) - max(x_min, x_min_gold)) * (
                    min(y_max, y_max_gold) - max(y_min, y_min_gold))
        inner_area = max(inner_area, 0)
        union_area = box_area + ground_truth_area - inner_area
        return inner_area / union_area

    def compute_reward(self, prev_box, cur_box, ground_truth_boxes):
        IOU_prev, _ = self.find_closest_box(prev_box, ground_truth_boxes)
        IOU_cur, _ = self.find_closest_box(cur_box, ground_truth_boxes)
        if IOU_cur - IOU_prev > 0:
            return 1
        return 0

    def train(self):
        xmin = 0.0
        xmax = self.width
        ymin = 0.0
        ymax = self.height

        for i_episode in range(self.episodes):
            for idx, (image, boxes) in enumerate(self.train_dataloader):
                # BATCHSIZE=1 is fixed, ignore this index
                image, boxes = image.to(self.device), boxes.to(self.device).squeeze(0)

                # history of past 9 actions, each consists of a one hot vec
                # TODO: this initialization might be improper for the action, should be one hot
                self.actions_history = torch.ones((9, 9)).to(self.device)
                prev_state = self.get_state(image)  # bz x (81 + image feature size)
                original_coordinates = [xmin, xmax, ymin, ymax]
                prev_box = original_coordinates
                prev_image = image

                t = 0
                while True:
                    action = self.get_action(prev_state).item()

                    # update the bounding box and update boolean mask
                    cur_box, done = self.update_box(action, prev_box)
                    if done:
                        # trigger event has different reward
                        IOU, _ = self.find_closest_box(prev_box, boxes)
                        if IOU >= self.threshold:
                            reward = self.eta
                            self.memory.push(prev_state, action, None, reward)
                        break


                    # update the observed region (change image region)
                    cur_image = self.update_observed_region(prev_image, cur_box)
                    next_state = self.get_state(cur_image)
                    # compute reward
                    reward = self.compute_reward(prev_box, cur_box, boxes)
                    self.memory.push(prev_state, action, next_state, reward)

                    t += 1
                    if t == self.max_steps_per_episode:
                        break
                    # tracked object update for next loop
                    prev_box = cur_box
                    prev_image = cur_image
                    prev_state = next_state
                    self.train_policy_net(5)

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_policy_net(self, min_sample):
        if len(self.memory) < min_sample:
            return
        # optimize the model using backprop
        BATCH_SIZE=4
        sample_tuple = self.memory.sample(BATCH_SIZE) # TODO: hardcode bz=4 for training for now
        # TODO: maybe this construction step can be parallelized as well?
        rewards = self.actions_history.new_full((BATCH_SIZE, 1), 0)
        actions = self.actions_history.new_full((BATCH_SIZE, 1), 0).type(torch.int64).to(self.device)
        cur_state_list = []
        next_state_list = []
        final_state_mask = self.actions_history.new_full((BATCH_SIZE, 1), 0).squeeze()
        for i in range(BATCH_SIZE):
            cur_state, action, next_state, reward = sample_tuple[i]
            rewards[i] = reward
            actions[i] = action
            if next_state is not None:
                next_state_list.append(next_state)
            else:
                final_state_mask[i] = 1
                next_state_list.append(cur_state.new_full((1, cur_state.size(1)), 0))
            cur_state_list.append(cur_state)
        cur_state_tensor = Variable(torch.stack(cur_state_list, dim=0).squeeze().to(self.device))
        next_state_tensor = Variable(torch.stack(next_state_list, dim=0).squeeze().to(self.device))
        Q = self.policy_net(cur_state_tensor).gather(1, actions)

        next_state_action_value = self.target_net(next_state_tensor)
        max_next_state_action_value, _ = torch.max(next_state_action_value, dim=1)
        # print(max_next_state_action_value, 1-final_state_mask)
        masked_next_value = max_next_state_action_value * (1-final_state_mask)
        expected_Q = masked_next_value * self.gamma + rewards

        loss = self.criterion(Q, expected_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
