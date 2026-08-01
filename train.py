import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F

from model import Transformer, make_src_mask, make_tgt_mask, CFDViT

from tqdm import tqdm

from config.config import *

from Data import dataset_cfd
from Data import fourier_features

from model.lr_scheduler import *

import wandb
import json


def run_epoch(
    data_iter,
    model: CFDViT,
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
 
        for src, tgt, domain_mask in tqdm(data_iter):
 
            src = src.to(device)
            tgt = tgt.to(device)
            domain_mask = domain_mask.to(device)
 
            # Pure ViT: one parallel forward pass predicts every patch at once.
            # No shifting, no start token, no autoregression -- so there's no
            # exposure-bias gap between this loss and what you'll see at inference.
            logits = model(src)
 
            # loss
            loss = loss_fn(logits, tgt, src, domain_mask)
 
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
                if scheduler is not None:
                    scheduler.step()
 
            total_loss += loss.item()
        losses.append(total_loss / len(data_iter))
 
    return sum(losses)/len(losses)

def predict_field(model: CFDViT, src: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Inference for the pure ViT: a single forward pass predicts every patch
    at once. No loop, no autoregression -- this is the same computation
    that produces the training/validation loss, so there's no gap between
    reported loss and what you'll see when you plot the result.
 
    src : (B, 1) normalized Reynolds number
    returns : (B, num_patches, patch_dim) predicted field, in patch form
    """
    model.eval()
    with torch.no_grad():
        return model(src.to(device))

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

    checkpoint = torch.load(path, map_location=device, weights_only = False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]

        
class CFDLoss(nn.Module):
    def __init__(self, grad_weight=0.1, div_weight=0.1, patch_size=8, grid_size=64, channels=C, num_freq = FOURIER_FEATURES):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.grad_weight = grad_weight
        self.div_weight  = div_weight
        self.patch_size  = patch_size
        self.grid_size   = grid_size
        self.channels    = channels
        
        self.num_freq    = num_freq
 
    def patches_to_field(self, patches, channels):
        # patches: (B, num_patches, patch_dim) -> (B, C, H, W)
        B, num_patches, patch_dim = patches.shape
        p = int(num_patches ** 0.5)
        spatial = p * self.patch_size
 
        patches = patches.view(B, p, p, channels, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(B, channels, spatial, spatial)
 
    def forward(self, pred, target, re_norm, domain_mask):
        
        pred = pred[:, :, :]          # removing pos embedding
        target = target[:, :, :]      # removing pos embedding
        B, seq_len, patch_dim = pred.shape     
        p = self.grid_size // self.patch_size  # 16
        
        # trim to largest complete square that fits
        complete = (int(seq_len ** 0.5)) ** 2  # largest perfect square <= seq_len
        
        pred_field   = self.patches_to_field(pred[:, :complete, :], self.channels)
        target_field = self.patches_to_field(target[:, :complete, :-FOURIER_DIMENSIONS], self.channels)
        domain_mask  = self.patches_to_field(domain_mask, 1)
        domain_mask  = domain_mask.repeat(1, self.channels, 1, 1)
 
 
        sq = (pred_field - target_field).pow(2)
        sq = sq * domain_mask
        mse_loss = sq.sum() / domain_mask.sum()
    
        return (1 * mse_loss)
    
def run_training_experiment() -> None:

    # 2. Build dataset from dataset.py
    # 3. Create DataLoaders for train / val 


    cfd_dataset = dataset_cfd.CFD_Dataset(
        # root="Data_with_P",
        root="flow_past_cylinder_domain",
        patch_size = PATCH_SIZE, 
        grid_size  = GRID_SIZE

    )

    # split sizes
    train_size = int(TRAIN_SPLIT * len(cfd_dataset))
    test_size  = len(cfd_dataset) - train_size

    # random split
    train_dataset, test_dataset = torch.utils.data.random_split(
        cfd_dataset,
        [train_size, test_size]
    )

    # dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )

    train_re = []
    for src, _, _ in train_dataloader:
        train_re.extend([i.item() for i in (src * cfd_dataset.re_std) + cfd_dataset.re_mean])

    test_re = []
    for src, _, _ in test_dataloader:
        test_re.extend([i.item() for i in (src * cfd_dataset.re_std) + cfd_dataset.re_mean])

    
    data = {"train_re": train_re, "test_re": test_re}
    with open("train_test_re.json", "w", encoding="utf-8") as file:
        json.dump(data, file)


    # 1. Init W&B
    wandb.init(project="Machine Visiosn Transformer")

    # 4. Instantiate Transformer with hyperparameters from config    
    transformer = CFDViT(
                         d_model        = D_MODEL, 
                         N              = N, 
                         num_heads      = NUM_HEADS, 
                         d_ff           = D_FF, 
                         patch_dim      = PATCH_DIM - FOURIER_DIMENSIONS,
                         dropout        = DROPOUT)
    
    transformer = transformer.to(DEVICE)

    # 5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
    optimizer = optim.Adam(transformer.parameters(), betas = [0.9, 0.98], lr=1e-4)

    # 6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
    scheduler = NoamScheduler(optimizer, d_model = D_MODEL, warmup_steps = 5000, const_lr=True)

    # 7. Instantiate MSE Loss or smthing idk
    # loss_fn = torch.nn.MSELoss()
    loss_fn = CFDLoss(patch_size = PATCH_SIZE, grid_size = GRID_SIZE)

    # 8. Training loop:
    for epoch in range(EPOCHS):
        transformer.train()
        train_loss = run_epoch(train_dataloader, transformer, loss_fn,
                        optimizer, scheduler, 1, is_train=True, device=DEVICE)
        transformer.eval()
        test_loss = run_epoch(test_dataloader, transformer, loss_fn,
                        optimizer, scheduler, 1, is_train=False, device=DEVICE)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})
        print(f"EPOCH: {epoch} => Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")
        
        if (epoch % SAVE_EVERY == 0) or (epoch == EPOCHS-1):
            print(f"Saving at epoch: {epoch}")
            save_checkpoint(transformer, optimizer, scheduler, epoch)
    
    return transformer