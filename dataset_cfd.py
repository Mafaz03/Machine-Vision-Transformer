import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pathlib
import cv2
import pandas as pd
from scipy.interpolate import griddata
import os

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

            # normalize x, y to [0, 1] for consistent interpolation
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)

            points = np.stack([x, y], axis=1)  # (N, 2)

            # interpolate u and v onto regular grid
            u_grid = griddata(points, u, (grid_x, grid_y), method="linear", fill_value=0.0)
            v_grid = griddata(points, v, (grid_x, grid_y), method="linear", fill_value=0.0)

            # stack into (C, H, W) with C=2 (u and v channels)
            uv_grid = np.stack([u_grid, v_grid], axis=0).astype(np.float32)  # (2, 64, 64)

            self.u_mean_list.append(uv_grid[0].mean())
            self.u_std_list.append(uv_grid[0].std())
            self.v_mean_list.append(uv_grid[1].mean())
            self.v_std_list.append(uv_grid[1].std())

            self.re_list.append(re_value)
            self.patches_list.append(patches)
        
        self.re_mean = np.mean(self.re_list)
        self.re_std  = np.std(self.re_list)

        self.u_mean = np.mean(self.u_mean_list)
        self.u_std  = np.mean(self.u_std_list)
        self.v_mean = np.mean(self.v_mean_list)
        self.v_std  = np.mean(self.v_std_list)

        for i, uv_grid in enumerate(self.patches_list):
            uv_grid[0] = (uv_grid[0] - self.u_mean) / (self.u_std + 1e-8)
            uv_grid[1] = (uv_grid[1] - self.v_mean) / (self.v_std + 1e-8)

            uv_tensor = torch.tensor(uv_grid)
            patches = uv_tensor.unsqueeze(0)
            patches = patches.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5)
            _, pr, pc, C, ph, pw = patches.shape
            patches = patches.contiguous().view(pr * pc, C * ph * pw)
            self.patches_list[i] = patches


    def __len__(self): return len(self.re_list)
    
    def __getitem__(self, index):
        re_value = self.re_list[index]

        # normalize Reynolds number
        re_norm = (re_value - self.re_mean) / self.re_std
        re_tensor = torch.tensor([re_norm], dtype=torch.float32)

        return (re_tensor, self.patches_list[index])


if "__main__" == __name__:
    cfd_dataset = CFD_Dataset(root = "Data", patch_size=16, grid_size=64)
    dataloader  = DataLoader(cfd_dataset, batch_size = 1, shuffle = True)

    re, patches = next(iter(dataloader))

    print("src:", re.shape)   # (B, 1)
    print("tgt:", patches.shape)  # (B, 16, 512)  -- 16 patches, 2*16*16=512 patch_dim
    
    patches = patches.squeeze(0) # remove B for now
    re = re.squeeze(0) # remove B for now

    unrolled = patches.view(64//16, 64//16, 2, 16, 16)              # (patch_row, patch_col, C, patch_h, patch_w)
    unrolled = unrolled.permute(2, 0, 3, 1, 4).contiguous()         # (C, patch_row, patch_h, patch_col, patch_w)
    unrolled = unrolled.view(2, 64, 64)                             # (2, grid_size, grid_size)

    u = unrolled[0]                                                 # (grid_size, grid_size)
    v = unrolled[1]                                                 # (grid_size, grid_size)

    x_grid = torch.linspace(0, 1, 64)
    y_grid = torch.linspace(0, 1, 64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].contourf(x_grid, y_grid, u, levels=50, cmap="viridis")
    axes[0].set_title("u velocity")
    axes[1].contourf(x_grid, y_grid, v, levels=50, cmap="viridis")
    axes[1].set_title("v velocity")
    plt.suptitle(f"Re (normalised): {re.item()}")
    plt.tight_layout()
    plt.show()