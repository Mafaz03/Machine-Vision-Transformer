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
            B, seq_len, patch_dim = tgt.shape
            start_token = torch.zeros(B, 1, patch_dim).to(device)
            tgt_input = torch.cat([start_token, tgt[:, :-1, :]], dim=1) # [B,]
            tgt_output = tgt

            # tgt_input = tgt[:, :-1, :]
            # tgt_output = tgt[:, 1:, :]

            tgt_mask = make_tgt_mask(tgt_input)

            # forward
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # loss
            loss = loss_fn(logits, tgt_output, src)

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


def greedy_decode(model, src, src_mask, max_len, patch_dim, coords_tensor, num_freq = 16, device = "cpu"):
    # coords_tensor: [max_len, 2], where max_len: (grid_size // patch_size)**2

    pos_dim = 4 * num_freq  # 32

    with torch.no_grad():
        src = src.to(device)
        src_mask = src_mask.to(device)

        B = src.shape[0]
        # encode
        ys = torch.zeros(B, 1, patch_dim).to(device)
        memory = model.encode(src, src_mask)

        # loop
        for i in tqdm(range(max_len)):
            # inject known coords for current position
            # coords_so_far = coords_tensor[:ys.shape[1]].unsqueeze(0).expand(B, -1, pos_dim).to(device) # [B, i, pos_dim]
            # ys_with_coords = torch.cat([ys, coords_so_far], dim=-1)                                    # [B, i, i + pos_dim]

            start_coord = torch.zeros(1, pos_dim).to(device)
            coords_so_far = coords_tensor[:ys.shape[1]-1].to(device)
            coords_so_far = torch.cat([start_coord, coords_so_far], dim=0)                              # (seq_len, pos_dim)
            coords_so_far = coords_so_far.unsqueeze(0).expand(B, -1, pos_dim)                           # [B, i, pos_dim]
            ys_with_coords = torch.cat([ys, coords_so_far], dim=-1)                                     # [B, i, i + pos_dim]

            # decoding
            tgt_mask = make_tgt_mask(ys_with_coords).to(device)
            out = model.decode(
                    memory,
                    src_mask,
                    ys_with_coords,
                    tgt_mask
                ) # [B, num_patches, patch_dim]

            # newest predicted patch
            # next_patch = out[:, -1:, :]
            next_patch = out[:, -1:, :-pos_dim]  # strip coord dims from output

            # append
            ys = torch.cat([ys, next_patch], dim=1)

    return ys[:, 1:, :] # (B, num_patches, C*ph*pw) -- no coords



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

# class CFDLoss(nn.Module):
#     def __init__(self, grad_weight=0.1):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.grad_weight = grad_weight

#     def gradient_loss(self, pred, target):
#         # pred, target: (B, seq, patch_dim)
#         pred_grad   = torch.diff(pred,   dim=1)
#         target_grad = torch.diff(target, dim=1)
#         return self.mse(pred_grad, target_grad)

#     def forward(self, pred, target):
#         mse_loss  = self.mse(pred, target)
#         grad_loss = self.gradient_loss(pred, target)
#         return mse_loss + self.grad_weight * grad_loss
        
class CFDLoss(nn.Module):
    def __init__(self, grad_weight=0.1, div_weight=0.1, patch_size=8, grid_size=64, channels=2, num_freq = 16):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.grad_weight = grad_weight
        self.div_weight  = div_weight
        self.patch_size  = patch_size
        self.grid_size   = grid_size
        self.channels    = channels
        
        self.num_freq    = num_freq

    def patches_to_field(self, patches):
        # patches: (B, num_patches, patch_dim) -> (B, C, H, W)
        B, num_patches, patch_dim = patches.shape
        p = int(num_patches ** 0.5)
        spatial = p * self.patch_size

        patches = patches.view(B, p, p, self.channels, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(B, self.channels, spatial, spatial)

    def spatial_gradient_loss(self, pred, target):
        # pred, target: (B, C, H, W)
        # x gradients
        pred_dx   = torch.diff(pred,   dim=3)  # along W
        target_dx = torch.diff(target, dim=3)
        # y gradients
        pred_dy   = torch.diff(pred,   dim=2)  # along H
        target_dy = torch.diff(target, dim=2)

        return self.mse(pred_dx, target_dx) + self.mse(pred_dy, target_dy)

    def divergence_loss(self, pred):
        # pred: (B, C, H, W) where C=0 is u, C=1 is v
        # continuity equation: du/dx + dv/dy = 0
        du_dx = torch.diff(pred[:, 0, :, :], dim=2)  # (B, H, W-1)
        dv_dy = torch.diff(pred[:, 1, :, :], dim=1)  # (B, H-1, W)

        # match sizes
        min_h = min(du_dx.shape[1], dv_dy.shape[1])
        min_w = min(du_dx.shape[2], dv_dy.shape[2])

        divergence = du_dx[:, :min_h, :min_w] + dv_dy[:, :min_h, :min_w]
        return divergence.pow(2).mean()  # should be 0 (best case)
    
    def maginitude_loss(self, pred, target):
        # pred: (B, C, H, W) where C=0 is u, C=1 is v

        u_pred = pred[:, 0, :, :]
        v_pred = pred[:, 1, :, :]

        u_target = target[:, 0, :, :]
        v_target = target[:, 1, :, :]

        return ((((u_pred**2) + (v_pred**2)) ** 0.5) - (((u_target**2) + (v_target**2)) ** 0.5)).mean().abs()


    def forward(self, pred, target, re_norm):
        pos_dim = 4 * self.num_freq  # 32
        pred = pred[:, :, :-pos_dim]          # removing pos embedding
        target = target[:, :, :-pos_dim]      # removing pos embedding
        B, seq_len, patch_dim = pred.shape     
        p = self.grid_size // self.patch_size  # 16
        
        # trim to largest complete square that fits
        complete = (int(seq_len ** 0.5)) ** 2  # largest perfect square <= seq_len
        
        pred_field   = self.patches_to_field(pred[:, :complete, :])
        target_field = self.patches_to_field(target[:, :complete, :])


        mse_loss  = self.mse(pred_field, target_field)
        grad_loss = self.spatial_gradient_loss(pred_field, target_field)
        div_loss  = self.divergence_loss(pred_field)  # physics constraint
        mag_loss  = self.maginitude_loss(pred_field, target_field)  

        re_weight = 1 + torch.exp(-re_norm).mean() # smaller re batch -> more importance

        return (1 * mse_loss * re_weight) + (1 * mag_loss)# + (self.grad_weight * grad_loss) + (self.div_weight * div_loss)
    
def run_training_experiment() -> None:

    # config = {
    #     "grid_size"        : 64,
    #     "patch_size"       : 4,
    #     "patch_dim"        : 4 * 4 * 2,
    #     "d_model"          : 512,
    #     "N"                : 10,
    #     "num_heads"        : 32,
    #     "d_ff"             : 1024,
    #     "dropout"          : 0.1,
    #     "train_batch_size" : 2,
    #     "test_batch_size"  : 10,
    #     "epochs"           : 80,
    #     "device"           : 'cuda' if torch.cuda.is_available() else 'cpu',
    #     'save_every'       : 4
    # }

    # small
    config = {
        "grid_size"        : 64,
        "patch_size"       : 8,    
        "patch_dim"        : (8*8*2) + (4 * 16), # +64 because positional embedding was done in the dataset itself
        "d_model"          : 256,
        "N"                : 6,
        "num_heads"        : 8,    
        "d_ff"             : 1024,
        "dropout"          : 0.01,
        "train_batch_size" : 8,
        "test_batch_size"  : 8,
        "epochs"           : 500,
        "device"           : 'cuda' if torch.cuda.is_available() else 'cpu',
        "save_every"       : 20
    }
    # 2. Build dataset from dataset.py
    # 3. Create DataLoaders for train / val 


    cfd_dataset = CFD_Dataset(
        root="Data",
        patch_size = config["patch_size"], 
        grid_size = config["grid_size"]

    )

    # split sizes
    train_size = int(0.9 * len(cfd_dataset))
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
                              d_model        = config["d_model"], 
                              N              = config["N"], 
                              num_heads      = config["num_heads"], 
                              d_ff           = config["d_ff"], 
                              patch_dim      = config['patch_dim'])
    
    transformer = transformer.to(config["device"])

    # 5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
    optimizer = optim.Adam(transformer.parameters(), betas = [0.9, 0.98], lr=1e-4)

    # 6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
    scheduler = NoamScheduler(optimizer, d_model = config["d_model"], warmup_steps = 5000, const_lr=True)

    # 7. Instantiate MSE Loss or smthing idk
    # loss_fn = torch.nn.MSELoss()
    loss_fn = CFDLoss(patch_size = config['patch_size'], grid_size = config['grid_size'])



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