import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F

from model import Transformer, make_src_mask, make_tgt_mask

from tqdm import tqdm

from dataset_cfd import *
from lr_scheduler import *

import wandb


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:

    model.train() if is_train else model.eval()
    losses = []
    for _ in range(epoch_num):
        total_loss = 0

        for src, tgt in tqdm(data_iter):

            src = src.to(device)
            tgt = tgt.to(device)

            # masks
            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt)

            # shift for teacher forcing
            tgt_input = tgt[:, :-1, :]
            tgt_output = tgt[:, 1:, :]

            tgt_mask = make_tgt_mask(tgt_input)

            # forward
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # loss
            loss = loss_fn(logits, tgt_output)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            # import pdb; pdb.set_trace()
            total_loss += loss.item()
        losses.append(total_loss / len(data_iter))

    return sum(losses)/len(losses)


def greedy_decode(model, src, src_mask, max_len, patch_dim, device = "cpu"):
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    B = src.shape[0]
    # encode
    ys = torch.zeros(B, 1, patch_dim).to(device)
    memory = model.encode(src, src_mask)

    # loop
    for _ in range(max_len):
        # decoding
        tgt_mask = make_tgt_mask(ys).to(device)
        out = model.decode(
                memory,
                src_mask,
                ys,
                tgt_mask
            )
        
        # newest predicted patch
        next_patch = out[:, -1:, :]

        # append
        ys = torch.cat([ys, next_patch], dim=1)

    return ys[:, 1:, :]


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": {
                "d_model"       : model.d_model,
                "N"             : model.N,
                "num_heads"     : model.num_heads,
                "d_ff"          : model.d_ff,
                "dropout"       : model.dropout,
            }
         }
    ,path
    )

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device = "cpu"
) -> int:

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]


def run_training_experiment() -> None:

    config = {
        "grid_size"        : 128,
        "patch_size"       : 4,
        "patch_dim"        : 4 * 4 * 2,
        "d_model"          : 256,
        "N"                : 8,
        "num_heads"        : 8,
        "d_ff"             : 1024,
        "dropout"          : 0.1,
        "train_batch_size" : 20,
        "test_batch_size"  : 10,
        "epochs"           : 100,
        "device"           : 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every'       : 4
    }

    # 2. Build dataset from dataset.py
    # 3. Create DataLoaders for train / val 


    cfd_dataset = CFD_Dataset(
        root="Data",
        patch_size=config["patch_size"], 
        grid_size = config["grid_size"]

    )

    # split sizes
    train_size = int(0.8 * len(cfd_dataset))
    test_size  = len(cfd_dataset) - train_size

    # random split
    train_dataset, test_dataset = torch.utils.data.random_split(
        cfd_dataset,
        [train_size, test_size]
    )

    # dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False
    )

    # 1. Init W&B
    wandb.init(project="Machine Visiosn Transformer", config = config)

    # 4. Instantiate Transformer with hyperparameters from config
    transformer = Transformer(
                            #   src_vocab_size = config["src_vocab_size"], 
                              d_model        = config["d_model"], 
                              N              = config["N"], 
                              num_heads      = config["num_heads"], 
                              d_ff           = config["d_ff"], 
                              patch_dim      = config['patch_dim'])
    
    transformer = transformer.to(config["device"])

    # 5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
    optimizer = optim.Adam(transformer.parameters(), betas = [0.9, 0.98], lr=1)

    # 6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
    scheduler = NoamScheduler(optimizer, d_model = config["d_model"], warmup_steps = 5000)

    # 7. Instantiate MSE Loss or smthing idk
    loss_fn = torch.nn.MSELoss()

    # 8. Training loop:
    for epoch in range(config['epochs']):
        transformer.train()
        train_loss = run_epoch(train_dataloader, transformer, loss_fn,
                        optimizer, scheduler, 1, is_train=True, device=config['device'])
        transformer.eval()
        test_loss = run_epoch(test_dataloader, transformer, loss_fn,
                        optimizer, scheduler, 1, is_train=False, device=config['device'])
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})
        print(f"EPOCH: {epoch} => Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")
        
        if (epoch % config['save_every'] == 0) or (epoch == config["epochs"]-1):
            print(f"Saving at epoch: {epoch}")
            save_checkpoint(transformer, optimizer, scheduler, epoch)
    
    return transformer