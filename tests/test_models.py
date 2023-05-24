from hydra import initialize, compose
import torch

from brain2face.models import Classifier

with initialize(version_base=None, config_path="../configs/"):
    args = compose(config_name="ylab_ecog.yaml")


def test_classifier():
    classifier = Classifier(args)

    Y = torch.rand(64, 512, 90)
    Z = torch.rand(64, 512, 90)

    _, _, similarity_train = classifier(Z, Y)
    _, _, similarity_test = classifier(Z, Y, sequential=True)

    assert torch.allclose(similarity_train, similarity_test)
