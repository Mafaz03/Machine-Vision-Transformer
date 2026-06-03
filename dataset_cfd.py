import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pathlib
import cv2
import pandas as pd
from scipy.interpolate import griddata
import os
from config import *


def fourier_features(cords: torch.tensor, num_freq = 8):
    # [num_patches, 2] -> [num_patches, C * num_freq * 2]
    freqs = 2 ** torch.linspace(0, num_freq - 1, num_freq)                   # [num_freqs]
    angles = (cords.unsqueeze(-1) * freqs * C * torch.pi)                    # [num_patches, 2, num_freq]
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim = -1)    # [num_patches, 2, num_freq * 2]
    return encoded.view(cords.shape[0], -1)                                  # [num_patches, 2 * num_freq * 2]



class CFD_Dataset(Dataset):

    def __init__(self, root: str = "Data", patch_size: int = 8, grid_size = 64):
        super().__init__()

        root = pathlib.Path(root)

        self.re_list      = []
        self.text_list    = []
        self.patches_list = []

        self.u_mean_list = []
        self.u_std_list  = []
        self.v_mean_list = []
        self.v_std_list  = []
        self.P_mean_list = []
        self.P_std_list  = []

        
        files = os.listdir(root)

        # regular grid to interpolate onto
        lin = np.linspace(0, 1, grid_size)
        grid_x, grid_y = np.meshgrid(lin, lin)  # (grid_size, grid_size)

        for file in files:
            if not file.endswith(".csv"):
                continue

            # extract Re from filename e.g. "Re_100.csv"
            re_value = float(file.split("Re_")[-1].replace(".csv", ""))

            df = pd.read_csv(pathlib.Path(root) / file, index_col=0)
            df = df.dropna()

            x = df["x"].values.astype(np.float32)
            y = df["y"].values.astype(np.float32)
            u = df["u (m/s)"].values.astype(np.float32)
            v = df["v (m/s)"].values.astype(np.float32)
            P = df["p (Pa)"].values.astype(np.float32)

            # normalize x, y to [0, 1] for consistent interpolation
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            P = (P - P.min()) / (P.max() - P.min() + 1e-8)

            points = np.stack([x, y], axis=1)  # (N, 3)

            # interpolate u, v, P onto regular grid
            u_grid = griddata(points, u, (grid_x, grid_y), method="linear", fill_value=0.0)
            v_grid = griddata(points, v, (grid_x, grid_y), method="linear", fill_value=0.0)
            P_grid = griddata(points, P, (grid_x, grid_y), method="linear", fill_value=0.0)

            # stack into (C, H, W) with C=3 (u, v, P channels)
            uv_grid = np.stack([u_grid, v_grid, P_grid], axis=0).astype(np.float32)  # (3, 64, 64)

            self.u_mean_list.append(uv_grid[0].mean())
            self.u_std_list.append(uv_grid[0].std())
            self.v_mean_list.append(uv_grid[1].mean())
            self.v_std_list.append(uv_grid[1].std())
            self.P_mean_list.append(uv_grid[2].mean())
            self.P_std_list.append(uv_grid[2].std())

            self.re_list.append(re_value)
            self.patches_list.append(uv_grid)
        
        self.re_mean = np.mean(self.re_list)
        self.re_std  = np.std(self.re_list)
        self.u_mean = np.mean(self.u_mean_list)
        self.u_std  = np.mean(self.u_std_list)
        self.v_mean = np.mean(self.v_mean_list)
        self.v_std  = np.mean(self.v_std_list)
        self.P_mean = np.mean(self.P_mean_list)
        self.P_std  = np.mean(self.P_std_list)

        # postional embedding from file itself
        coords = []
        patches_per_side = grid_size // patch_size
        for row in range(patches_per_side):
            for col in range(patches_per_side):
                cx = (col + 0.5) / patches_per_side  # normalized [0,1]
                cy = (row + 0.5) / patches_per_side
                coords.append([cx, cy])

        coords_tensor = torch.tensor(coords, dtype=torch.float32)                # (num_patches, C)
        coords_tensor = fourier_features(cords = coords_tensor, num_freq = 16)   # (num_patches, C * 16 * 2)

        for i, uv_grid in enumerate(self.patches_list):
            uv_grid[0] = (uv_grid[0] - self.u_mean) / (self.u_std + 1e-8)
            uv_grid[1] = (uv_grid[1] - self.v_mean) / (self.v_std + 1e-8)
            uv_grid[2] = (uv_grid[2] - self.P_mean) / (self.P_std + 1e-8)

            uv_tensor = torch.tensor(uv_grid)
            patches = uv_tensor.unsqueeze(0)                                                        # (1, grid_size, grid_size), grid_size: actual size
            patches = patches.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)   # (1, C, patch_row, patch_col, patch_h, patch_w)
            patches = patches.permute(0, 2, 3, 1, 4, 5)                                             # (1, patch_row, patch_col, C, patch_h, patch_w)
            _, pr, pc, C, ph, pw = patches.shape
            patches = patches.contiguous().view(pr * pc, C * ph * pw)                               # patches: (patch_row * patch_col, C * patch_h * patch_w)
            patches = torch.cat([patches, coords_tensor], dim=-1)                                   # patches: (patch_row * patch_col, C * patch_h * patch_w + (2 * 2 * num_freq))
            self.patches_list[i] = patches


    def __len__(self): return len(self.re_list)
    
    def __getitem__(self, index):
        re_value = self.re_list[index]

        # normalize Reynolds number
        re_norm = (re_value - self.re_mean) / self.re_std
        re_tensor = torch.tensor([re_norm], dtype=torch.float32)

        return (re_tensor, self.patches_list[index])


if "__main__" == __name__:
    cfd_dataset = CFD_Dataset(root = "Data_with_P", patch_size = 16, grid_size = 64)
    dataloader  = DataLoader(cfd_dataset, batch_size = 1, shuffle = True)

    print("mean:   ", cfd_dataset.re_mean)
    print("std:    ", cfd_dataset.re_std)
    print("u_mean: ", cfd_dataset.u_mean)
    print("u_std:  ", cfd_dataset.u_std)
    print("v_mean: ", cfd_dataset.v_mean)
    print("v_std:  ", cfd_dataset.v_std)
    print("P_mean: ", cfd_dataset.P_mean)
    print("P_std:  ", cfd_dataset.P_std)

    re, patches = next(iter(dataloader))

    print("src:", re.shape)                                # (B, 1)
    print("tgt:", patches.shape)                           # (B, 16, 512 + (2 * C * 16))
    patches = patches[:, : , :-(2 * 2 * FOURIER_FEATURES)] # (B, 16, 512)
    print("tgt:", patches.shape)       # (B, 16, 512)      -- 16 patches, 2*16*16=512 patch_dim
    
    patches = patches.squeeze(0) # remove B for now
    re = re.squeeze(0)           # remove B for now

    unrolled = patches.view(64//16, 64//16, C, 16, 16)              # (patch_row, patch_col, C, patch_h, patch_w)
    unrolled = unrolled.permute(2, 0, 3, 1, 4).contiguous()         # (C, patch_row, patch_h, patch_col, patch_w)
    unrolled = unrolled.view(C, 64, 64)                             # (C, grid_size, grid_size)

    u = unrolled[0]                                                 # (grid_size, grid_size)
    v = unrolled[1]                                                 # (grid_size, grid_size)
    P = unrolled[2]                                                 # (grid_size, grid_size)

    x_grid = torch.linspace(0, 1, 64)
    y_grid = torch.linspace(0, 1, 64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].contourf(x_grid, y_grid, u, levels=50, cmap="viridis")
    axes[0].set_title("u velocity")
    axes[1].contourf(x_grid, y_grid, v, levels=50, cmap="viridis")
    axes[1].set_title("v velocity")
    axes[2].contourf(x_grid, y_grid, P, levels=50, cmap="viridis")
    axes[2].set_title("Pressure")
    plt.suptitle(f"Re (normalised): {re.item()}")
    plt.tight_layout()
    plt.show()