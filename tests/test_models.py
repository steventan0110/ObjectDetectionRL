import torch

from module.models import DuelingDQN


def test_dueling_DQN():
    BATCH_SIZE = 8
    model = DuelingDQN()
    model.train()
    features = torch.rand((BATCH_SIZE, 512*8*8+81))
    output = model(features)
    print('Output shape:', output.shape)


def run_all():
    test_dueling_DQN()
