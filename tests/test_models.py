from hydra import initialize, compose
import torch

from brain2face.models.classifier import Classifier
from brain2face.utils.loss import CLIPLoss

with initialize(version_base=None, config_path="../configs/"):
    args = compose(config_name="ylab_ecog.yaml")


def test_classifier():
    classifier = Classifier(args)

    Y = torch.rand(64, 512, 90)
    Z = torch.rand(64, 512, 90)

    _, _, similarity_train = classifier(Z, Y)
    _, _, similarity_test = classifier(Z, Y, sequential=True)

    assert torch.allclose(similarity_train, similarity_test)


def test_clip_loss():
    loss_func = CLIPLoss(args).eval()
    
    Y = torch.rand(64, 512, 90)
    Z = torch.rand(64, 512, 90)
    
    loss = loss_func(Y, Z)
    loss_slow = loss_func(Y, Z, fast=False)
    
    assert torch.allclose(loss, loss_slow)