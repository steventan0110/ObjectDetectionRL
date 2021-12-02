import os
import torch

from module.models import DQN, FeatureExtractor
from module.memory import ReplayMemory
from torch.autograd import Variable
from tqdm import tqdm
import random


class Agent:

    def __init__(self, cls, train_dataloader, valid_dataloader, alpha=0.2, threshold=0.5,
                 eta=3, epochs=15, max_steps_per_episode=20, **kwargs):

        # Defining constants
        self.cls = cls
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = 9
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.epochs = epochs
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval = kwargs['save_interval']
        self.save_dir = kwargs['save_dir']

        # Defining hyperparameters
        self.gamma = 0.9
        self.alpha = alpha
        self.threshold = threshold
        self.eta = eta
        self.epsilon = 1
        self.batch_size = kwargs['batch_size'] # Used when sampling from replay memory
        self.lr = kwargs['learning_rate']
        self.target_update = kwargs['target_update']

        # Initializing models
        self.extractor = FeatureExtractor(kwargs['image_extractor'], freeze=True).to(self.device)
        self.extractor.eval()  # do not update the pretrained CNN

        rl_algo = kwargs['rl_algo']
        if rl_algo == 'DQN':
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)  # DQN use two policy network
        elif rl_algo == 'DoubleDQN':
            raise ValueError('Not yet implemented')
        else:
            raise ValueError('Please select a reinforcement learning model to use')

        if kwargs['load_dir'] is not None:
            self.load_checkpoint(kwargs['load_dir'])
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Initializing remaining deep learning objects
        self.memory = ReplayMemory(10000)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def get_state(self, image):
        if len(image.shape) == 3:
            # Need to add batch dimension
            image = image.unsqueeze(0)
        image_feature = self.extractor(image)
        image_feature = image_feature.reshape(1, -1)
        history = self.actions_history.reshape(1, -1).type(torch.FloatTensor).to(self.device)
        state = torch.cat((image_feature, history), 1)
        return state

    def get_action_eval(self, state):
        with torch.no_grad():
            pred = self.policy_net(state)
            _, best_action = torch.max(pred, 1)
            return best_action

    def get_action(self, state, gt_boxes, actions):
        p = torch.rand(1).item()
        if p > self.epsilon:
            # exploitation
            pred = self.policy_net(state)
            _, best_action = torch.max(pred, 1)
            return best_action.item()
        else:
            # guided exploration instead of randomly choosing action in epsilon-greedy policy
            prev_box = self.update_box(actions)
            good_action = []
            bad_action = []
            for candidate_action in range(9):
                temp = actions.copy()
                temp.append(candidate_action)
                cur_box = self.update_box(temp)
                if candidate_action == 8: # terminate
                    IOU, _best_box = self.find_closest_box(cur_box, gt_boxes)
                    reward = self.eta if IOU >= self.threshold else -1 * self.eta
                else:
                    reward = self.compute_reward(prev_box, cur_box, gt_boxes)
                if reward >= 0:
                    good_action.append(candidate_action)
                else:
                    bad_action.append(candidate_action)
            if len(good_action) == 0:
                return random.choice(bad_action)
            return random.choice(good_action)


    def update_action_history(self, action):
        # Update action history
        # Always rolls the history so that the earliest action is no longer tracked
        # Note that action history is initialize to tensor of ones; the ones will be
        # the first to get rolled off

        # self.action_history has shape (number of actions, size of action space)
        self.actions_history = torch.roll(self.actions_history, 1, 0)
        action_one_hot = torch.zeros(self.num_actions, device=self.device)
        action_one_hot[action] = 1
        self.actions_history[0] = action_one_hot

    def update_box(self, actions):
        x_min, x_max, y_min, y_max = 0, self.width, 0, self.height
        alpha_h = self.alpha * (y_max - y_min)
        alpha_w = self.alpha * (x_max - x_min)

        for action in actions:
            if action == 0:  # left
                x_min -= alpha_w
                x_max -= alpha_w
            elif action == 1:  # right
                x_min += alpha_w
                x_max += alpha_w
            elif action == 2:  # up
                y_min -= alpha_h
                y_max -= alpha_h
            elif action == 3:  # down
                y_min += alpha_h
                y_max += alpha_h
            elif action == 4:  # bigger
                x_min -= alpha_w
                x_max += alpha_w
                y_min -= alpha_h
                y_max += alpha_h
            elif action == 5:  # smaller
                x_min += alpha_w
                x_max -= alpha_w
                y_min += alpha_h
                y_max -= alpha_h
            elif action == 6:  # shorter
                y_min += alpha_h
                y_max -= alpha_h
            elif action == 7:  # narrower
                x_min += alpha_w
                x_max -= alpha_w


        x_min = min(max(x_min, 0), self.width)
        x_max = min(max(x_max, 0), self.width)
        y_min = min(max(y_min, 0), self.height)
        y_max = min(max(y_max, 0), self.height)

        new_box = [x_min, x_max, y_min, y_max]
        return new_box

    def update_observed_region(self, image, box):
        from util.transforms import resize
        x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_roi = image[:, x_min:x_max, y_min:y_max]
        transform = resize(self.height, self.width)
        image, _box = transform(img_roi, [box])
        return image

    def find_closest_box(self, box, gt_boxes):
        max_area = -float('inf')
        best_box = None
        for gt_box in gt_boxes:
            area = self.compute_IOU(box, gt_box)
            if area > max_area:
                max_area = area
                best_box = gt_box
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
        IOU_prev, _best_box = self.find_closest_box(prev_box, ground_truth_boxes)
        IOU_cur, _best_box = self.find_closest_box(cur_box, ground_truth_boxes)
        if IOU_cur > IOU_prev:
            return torch.tensor(1, device=self.device)
        return torch.tensor(0, device=self.device)

    def train(self):
        xmin = 0.0
        xmax = self.width
        ymin = 0.0
        ymax = self.height

        self.policy_net.train()
        for epoch_i in range(1, self.epochs + 1):
            print(f'EPOCH {epoch_i} for class {self.cls}')
            training_loss = []
            for image, boxes in tqdm(self.train_dataloader):
                assert image.shape[0] == 1 and boxes.shape[0] == 1  # Batch size must be 1
                image, boxes = image.squeeze(0).to(self.device), boxes.squeeze(0).to(self.device)
                # Action history is initialized to all ones when episode first starts
                # This allows us to have a fix number of states when passing the features to the Q function
                # Real actions will be represented with a one hot vector
                # Can also try using all zeros.
                self.actions_history = torch.ones((9, 9), device=self.device)

                original_image = image.clone()
                prev_state = self.get_state(image)  # bz x (81 + image feature size)
                prev_box = [xmin, xmax, ymin, ymax]
                done = False
                # TODO is there a need to track all actions?
                all_actions = [] # track all previous action to update box

                t = 0
                while not done and t < self.max_steps_per_episode:
                    action = self.get_action(prev_state, boxes, all_actions)
                    done = action == 8
                    all_actions.append(action)

                    self.update_action_history(action)
                    cur_box = self.update_box(all_actions)

                    if done:
                        # Termination reward is different than reward from other actions
                        IOU, _best_box = self.find_closest_box(cur_box, boxes)
                        reward = self.eta if IOU >= self.threshold else -1 * self.eta
                        reward = torch.tensor(reward, device=self.device)
                        self.memory.push(prev_state, action, None, reward)
                    else:
                        x_min, x_max, y_min, y_max = int(cur_box[0]), int(cur_box[1]), int(cur_box[2]), int(cur_box[3])
                        if x_min >= x_max or y_min >= y_max:
                            break

                        cur_image = self.update_observed_region(original_image, cur_box)

                        next_state = self.get_state(cur_image)
                        reward = self.compute_reward(prev_box, cur_box, boxes)
                        self.memory.push(prev_state, action, next_state, reward)

                        # Tracked object update for next loop
                        prev_box = cur_box
                        prev_state = next_state

                    loss = self.train_policy_net()
                    if loss is not None: training_loss.append(loss)
                    t += 1

            avg_batch_loss = sum(training_loss) / len(training_loss)
            print(f'Average batch loss: {avg_batch_loss}')

            if epoch_i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()
                for param in self.target_net.parameters():
                    param.requires_grad = False

            if epoch_i % self.save_interval == 0:
                self.save_checkpoint(epoch_i, self.save_dir)

            if epoch_i <= 5:
                self.epsilon -= 0.9/5 # anneal epsilon from 1 to 0.1 following the paper

            # validate
            precision = self.validate()
            print(f'precision: {precision}')
            self.policy_net.train()

    def train_policy_net(self):
        # Optimize the model using backprop

        if len(self.memory) < self.batch_size:
            return None

        samples_tuple = self.memory.sample(self.batch_size)
        rewards = torch.zeros((self.batch_size, 1), device=self.device)
        actions = torch.zeros((self.batch_size, 1), dtype=torch.int64, device=self.device)
        final_state_mask = torch.zeros(self.batch_size, device=self.device)
        cur_state_list, next_state_list = [], []
        for i in range(self.batch_size):
            cur_state, action, next_state, reward = samples_tuple[i]
            cur_state_list.append(cur_state[0])
            rewards[i] = reward
            actions[i] = action
            if next_state is not None:
                next_state_list.append(next_state[0])
            else:
                final_state_mask[i] = 1
                empty_state = torch.zeros(cur_state.size(1), device=self.device)
                next_state_list.append(empty_state)
        cur_state_tensor = Variable(torch.stack(cur_state_list, dim=0).to(self.device))
        next_state_tensor = Variable(torch.stack(next_state_list, dim=0).to(self.device))

        # Calculating predicted q values
        Q = self.policy_net(cur_state_tensor).gather(1, actions)

        # Calculating target q values
        next_state_action_value = self.target_net(next_state_tensor)
        max_next_state_action_value, _ = torch.max(next_state_action_value, dim=1)
        masked_next_value = max_next_state_action_value * (1 - final_state_mask)
        masked_next_value = masked_next_value.unsqueeze(1) # Has shape batch size x 1
        expected_Q = rewards + self.gamma * masked_next_value

        loss = self.criterion(Q, expected_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def _validate_find_box(self, image, boxes):
        xmin = 0.0
        xmax = self.width
        ymin = 0.0
        ymax = self.height
        assert image.shape[0] == 1
        image = image.squeeze(0).to(self.device)
        self.actions_history = torch.ones((9, 9), device=self.device)
        original_coordinates = [xmin, xmax, ymin, ymax]
        original_image = image.clone()
        prev_state = self.get_state(image)  # bz x (81 + image feature size)
        done = False
        all_actions = []  # track all previous action to update box

        t = 0
        while not done and t < self.max_steps_per_episode:
            action = self.get_action_eval(prev_state)
            done = action == 8
            all_actions.append(action)

            self.update_action_history(action)
            cur_box = self.update_box(all_actions)
            if done:
                return cur_box
            else:
                x_min, x_max, y_min, y_max = int(cur_box[0]), int(cur_box[1]), int(cur_box[2]), int(cur_box[3])
                if x_min >= x_max or y_min >= y_max:
                    break
                # try:
                cur_image = self.update_observed_region(original_image, cur_box)
                next_state = self.get_state(cur_image)
                prev_state = next_state
            t += 1
        return cur_box # non terminating case, should be a bad box


    def _eval(self, hyp, tgt):
        assert len(hyp) == len(tgt)
        sample_size = len(hyp)
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        out = dict()
        for threshold in thresholds:
            out[threshold] = {
                'tp': 0,
                'fp': 0
            }
        for i in range(sample_size):
            predicted_box = hyp[i]
            gt_boxes = tgt[i].squeeze(0)
            iou, _ = self.find_closest_box(predicted_box, gt_boxes)
            for threshold in thresholds:
                if iou >= threshold:
                    out[threshold]['tp'] += 1.0
                else:
                    out[threshold]['fp'] += 1.0
        prec = []
        for threshold in thresholds:
            prec.append(out[threshold]['tp']/sample_size)
        return prec


    def validate(self):
        self.policy_net.eval()
        hyp = []
        tgt = []
        # compute the precision and recall on validation set
        for image, boxes in tqdm(self.valid_dataloader):
            box = self._validate_find_box(image, boxes)
            hyp.append(box)
            tgt.append(boxes)
        precision = self._eval(hyp, tgt)
        return precision


    def save_checkpoint(self, epoch, dir):
        save_path = os.path.join(dir, 'checkpoint_' + self.cls + '_' + str(epoch) + '.pt')
        torch.save({
            'policy_network': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load_checkpoint(self, dir):
        # TODO given dir, need to find checkpoint of latest epoch
        checkpoint = torch.load(dir, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
