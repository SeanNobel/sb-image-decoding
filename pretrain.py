import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm, trange
from configs.args import args
from termcolor import cprint
import importlib
import wandb

from data.datasets import ArayaDrivingDataset
from utils.layout import ch_locations_2d

sys.path.append("speech-decoding")
models = importlib.import_module("speech-decoding.models")
losses = importlib.import_module("speech-decoding.utils.loss")

sys.path.append("vivit")
from vivit import ViViT


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    print(f"Setting random seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

run_dir = f"runs/{args.pretrain_name}/"
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

if args.wandb:
    wandb.config = {k: v for k, v in args.__dict__.items() if not k.startswith("__")}
    wandb.init(
        project="f2b_contrastive",
        config=wandb.config,
        save_code=True,
    )
    wandb.run.name = args.pretrain_name
    wandb.run.save()


# -----------------------
#       Dataloader
# -----------------------
if args.split == "shallow":
    dataset = ArayaDrivingDataset(args)

    train_size = int(dataset.X.shape[0] * args.train_ratio)
    test_size = dataset.X.shape[0] - train_size
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
else:
    train_set = ArayaDrivingDataset(args)
    test_set = ArayaDrivingDataset(args, train=False)

loader_args = {"drop_last": True, "num_workers": 4, "pin_memory": True}
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=args.batch_size, shuffle=False, **loader_args
)

# ---------------------
#        Models
# ---------------------
brain_encoder = models.BrainEncoder(
    args, num_subjects=train_set.num_subjects, layout_fn=ch_locations_2d
).to(device)

face_encoder = ViViT(
    image_size=args.image_size,
    patch_size=args.patch_size,
    num_frames=args.seq_len * args.fps,
    dim=args.F,
    depth=args.depth,
    in_channels=1,
).to(device)

classifier = models.Classifier(args)

# ---------------
#      Loss
# ---------------
loss_func = losses.CLIPLoss(args).to(device)

# ---------------------
#      Optimizers
# ---------------------
# NOTE: Brain optim optimizes loss temperature
optim_brain = torch.optim.Adam(
    list(brain_encoder.parameters()) + list(loss_func.parameters()), lr=args.lr
)
optim_face = torch.optim.Adam(face_encoder.parameters(), lr=args.lr)

if args.lr_scheduler == "cosine":
    sched_brain = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_brain, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    sched_face = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_face, T_max=args.epochs, eta_min=args.lr * 0.01
    )
elif args.lr_scheduler == "multistep":
    mlstns = [int(m * args.epochs) for m in args.lr_multistep_mlstns]
    sched_brain = torch.optim.lr_scheduler.MultiStepLR(
        optim_brain, milestones=mlstns, gamma=args.lr_step_gamma
    )
    sched_face = torch.optim.lr_scheduler.MultiStepLR(
        optim_face, milestones=mlstns, gamma=args.lr_step_gamma
    )
else:
    raise ValueError()


# -----------------------
#     Strat training
# -----------------------
min_test_loss = float("inf")

for epoch in range(args.epochs):
    train_losses = []
    test_losses = []
    train_top10_accs = []
    train_top1_accs = []
    test_top10_accs = []
    test_top1_accs = []
    inference_times = []

    brain_encoder.train()
    face_encoder.train()
    loss_func.train()
    for i, (X, Y, subject_idxs) in enumerate(tqdm(train_loader)):

        X, Y = X.to(device), Y.to(device)

        X_f = brain_encoder(X, subject_idxs)
        Y_f = face_encoder(Y)

        loss = loss_func(Y_f, X_f)

        with torch.no_grad():
            train_top1_acc, train_top10_acc = classifier(X_f, Y_f)

        train_losses.append(loss.item())
        train_top10_accs.append(train_top10_acc)
        train_top1_accs.append(train_top1_acc)

        optim_brain.zero_grad()
        optim_face.zero_grad()

        loss.backward()

        optim_brain.step()
        optim_face.step()

    brain_encoder.eval()
    face_encoder.eval()
    loss_func.eval()
    for X, Y, subject_idxs in tqdm(test_loader):

        X, Y = X.to(device), Y.to(device)

        with torch.no_grad():
            X_f = brain_encoder(X, subject_idxs)

            stime = time()
            Y_f = face_encoder(Y)
            inference_times.append(time() - stime)

            loss = loss_func(Y_f, X_f)

            test_top1_acc, test_top10_acc = classifier(X_f, Y_f)

        test_losses.append(loss.item())
        test_top10_accs.append(test_top10_acc)
        test_top1_accs.append(test_top1_acc)

    print(
        f"Epoch {epoch}/{args.epochs} | ",
        f"avg train loss: {np.mean(train_losses):.3f} | ",
        f"avg test loss: {np.mean(test_losses):.3f} | ",
        f"lr: {optim_brain.param_groups[0]['lr']:.5f}",
    )

    if args.wandb:
        performance_now = {
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "test_loss": np.mean(test_losses),
            "train_top10_acc": np.mean(train_top10_accs),
            "train_top1_acc": np.mean(train_top1_accs),
            "test_top10_acc": np.mean(test_top10_accs),
            "test_top1_acc": np.mean(test_top1_accs),
            "lrate": optim_brain.param_groups[0]["lr"],
            "temp": loss_func.temp.item(),
            "FaceEncoder avg inference time": np.mean(inference_times),
        }
        wandb.log(performance_now)

    sched_brain.step()
    sched_face.step()

    # Save models
    torch.save(face_encoder.state_dict(), run_dir + "face_encoder_last.pt")
    torch.save(brain_encoder.state_dict(), run_dir + "brain_encoder_last.pt")

    if np.mean(test_losses) < min_test_loss:
        cprint(f"New best. Saving models to {run_dir}", color="cyan")

        torch.save(face_encoder.state_dict(), run_dir + "face_encoder_best.pt")
        torch.save(brain_encoder.state_dict(), run_dir + "brain_encoder_best.pt")

        min_test_loss = np.mean(test_losses)
