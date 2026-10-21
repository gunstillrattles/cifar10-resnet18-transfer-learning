import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from model import ResNetTransferLearning  # noqa: E402


def test_output_shape_and_frozen_backbone():
    m = ResNetTransferLearning(pretrained=False).eval()
    assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 10)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable == ["resnet.fc.weight", "resnet.fc.bias"]   # учится только голова
