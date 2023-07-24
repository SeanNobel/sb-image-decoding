from hydra import initialize, compose
import torch
from termcolor import cprint

from brain2face.models.brain_encoder import SubjectBlockSA
from brain2face.models.classifier import Classifier
from brain2face.utils.loss import CLIPLoss
from brain2face.utils.layout import DynamicChanLoc2d

with initialize(version_base=None, config_path="../configs/ylab"):
    args = compose(config_name="god.yaml")


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
    
def test_subject_block_sa():
    """Only works in YLab server.
    Need to comment assert pad.sum() == 0 in brain_encoder
    """
    subject_idxs = torch.randint(0, 4, (64, ))
    subject_names = ["E0068", "E0072", "EJ0008", "EJ0011"]
    num_subjects = len(subject_names)
    
    subject_block_sa = SubjectBlockSA(
        args, num_subjects, DynamicChanLoc2d(args, subject_names)
    ).eval()
    
    X = torch.rand(64, 512, 90)
    
    X_batch = subject_block_sa(X, subject_idxs, subbatch=True)
    X_sequntial = subject_block_sa(X, subject_idxs, subbatch=False)
    
    assert torch.allclose(X_batch, X_sequntial, atol=1e-6)