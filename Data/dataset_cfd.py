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
from tqdm import tqdm

def fourier_features(cords: torch.tensor, num_freq = 8):
    # [num_patches, 2] -> [num_patches, num_freq * 2]
    freqs = 2 ** torch.linspace(0, num_freq - 1, num_freq)                   # [num_freqs]
    angles = (cords.unsqueeze(-1) * freqs  * torch.pi)                       # [num_patches, 2, num_freq]
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
        self.mask_list   = []

        if C == 3: self.P_mean_list = []
        if C == 3: self.P_std_list  = []

        
        files = os.listdir(f"{ROOT}/Data/Problems{root}")

        # regular grid to interpolate onto
        lin = np.linspace(0, 1, grid_size)
        grid_x, grid_y = np.meshgrid(lin, lin)  # (grid_size, grid_size)

        for file in tqdm(files):
            if not file.endswith(".csv"):
                continue

            # extract Re from filename e.g. "Re_100.csv"
            re_value = float(file.split("Re_")[-1].replace(".csv", ""))
            df = pd.read_csv(pathlib.Path(root) / file, index_col=0)

            n = 64

            X = df["x"].to_numpy().reshape(n, n)
            Y = df["y"].to_numpy().reshape(n, n)

            u_grid = df["u (m/s)"].to_numpy().reshape(n, n)
            v_grid = df["v (m/s)"].to_numpy().reshape(n, n)
            P_grid = df["p (Pa)"].to_numpy().reshape(n, n)

            mask = df["mask"].to_numpy().reshape(n, n)

            u_grid = np.nan_to_num(u_grid)  
            v_grid = np.nan_to_num(v_grid)  
            P_grid = np.nan_to_num(P_grid)  

            self.mask_list.append(mask)

            # stack into (C, H, W) with C=3 (u, v, P channels)
            uv_grid = np.stack([u_grid, v_grid, P_grid], axis=0).astype(np.float32)  # (3, 64, 64)

        
            self.u_mean_list.append(uv_grid[0].mean())
            self.u_std_list.append(uv_grid[0].std())
            self.v_mean_list.append(uv_grid[1].mean())
            self.v_std_list.append(uv_grid[1].std())
            if C == 3: 
                self.P_mean_list.append(uv_grid[2].mean())
                self.P_std_list.append(uv_grid[2].std())

            self.re_list.append(re_value)
            self.patches_list.append(uv_grid)

        # Computing global stats from raw grids
        all_u = np.concatenate([g[0][m].flatten() for g, m in zip(self.patches_list, self.mask_list)])
        all_v = np.concatenate([g[1][m].flatten() for g, m in zip(self.patches_list, self.mask_list) ])
        all_P = np.concatenate([g[2][m == 1].flatten() for g, m in zip(self.patches_list, self.mask_list) ])

        self.u_mean, self.u_std = all_u.mean(), all_u.std()
        self.v_mean, self.v_std = all_v.mean(), all_v.std()
        self.P_mean, self.P_std = all_P.mean(), all_P.std()

        
        self.re_mean = np.mean(self.re_list)
        self.re_std  = np.std(self.re_list)


        # postional embedding from file itself
        coords = []
        patches_per_side = grid_size // patch_size
        for row in range(patches_per_side):
            for col in range(patches_per_side):
                cx = (col + 0.5) / patches_per_side  # normalized [0,1]
                cy = (row + 0.5) / patches_per_side
                coords.append([cx, cy])

        coords_tensor = torch.tensor(coords, dtype=torch.float32)                # (num_patches, C)
        coords_tensor = fourier_features(cords = coords_tensor, num_freq = FOURIER_FEATURES)   # (num_patches, C * 16 * 2)

        for i, (uv_grid, mask) in enumerate(zip(self.patches_list, self.mask_list)):
            uv_grid[0] = (uv_grid[0] - self.u_mean) / (self.u_std + 1e-8)
            uv_grid[1] = (uv_grid[1] - self.v_mean) / (self.v_std + 1e-8)
            uv_grid[2] = (uv_grid[2] - self.P_mean) / (self.P_std + 1e-8)

            uv_tensor = torch.tensor(uv_grid)
            
            
            patches = uv_tensor.unsqueeze(0)                                                        # (1, grid_size, grid_size), grid_size: actual size
            patches = patches.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)   # (1, C, patch_row, patch_col, patch_h, patch_w)
            patches = patches.permute(0, 2, 3, 1, 4, 5)                                             # (1, patch_row, patch_col, C, patch_h, patch_w)
            _, pr, pc, num_ch, ph, pw = patches.shape
            patches = patches.contiguous().view(pr * pc, C * ph * pw)                               # patches: (patch_row * patch_col, C * patch_h * patch_w)
            patches = torch.cat([patches, coords_tensor], dim=-1)                                   # patches: (patch_row * patch_col, C * patch_h * patch_w + (2 * 2 * num_freq))
            self.patches_list[i] = patches


            mask_tensor  = torch.tensor(mask).unsqueeze(0)   # (1,H,W) 
            mask_patches = mask_tensor.unsqueeze(0)          # (1,1,H,W) 
            mask_patches = mask_patches.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            mask_patches = mask_patches.permute(0,2,3,1,4,5)
            _, pr, pc, _, ph, pw = mask_patches.shape
            mask_patches = mask_patches.contiguous().view(pr*pc, ph*pw)
            self.mask_list[i] = mask_patches


    def __len__(self): return len(self.re_list)
    
    def __getitem__(self, index):
        re_value = self.re_list[index]

        # normalize Reynolds number
        re_norm = (re_value - self.re_mean) / self.re_std
        re_tensor = torch.tensor([re_norm], dtype=torch.float32)

        return (re_tensor, self.patches_list[index], ~self.mask_list[index])


if "__main__" == __name__:
    # cfd_dataset = CFD_Dataset(root = "Data_with_P", patch_size = 16, grid_size = 64)
    
    cfd_dataset = CFD_Dataset(root = "Data/Problems/flow_past_cylinder_domain", patch_size = 16, grid_size = 64)
    dataloader  = DataLoader(cfd_dataset, batch_size = 1, shuffle = True)

    print("re mean: ", cfd_dataset.re_mean)
    print("re std:  ", cfd_dataset.re_std)
    print("u_mean:  ", cfd_dataset.u_mean)
    print("u_std:   ", cfd_dataset.u_std)
    print("v_mean:  ", cfd_dataset.v_mean)
    print("v_std:   ", cfd_dataset.v_std)
    
    if C == 3: 
        print("P_mean:  ", cfd_dataset.P_mean)
        print("P_std:   ", cfd_dataset.P_std)

    re, patches, mask = next(iter(dataloader))

    print("src:", re.shape)                                # (B, 1)
    print("tgt:", patches.shape)                           # (B, 16, 512 + (4 * 16))
    patches = patches[:, : , :-(2 * 2 * FOURIER_FEATURES)] # (B, 16, 512)
    print("tgt:", patches.shape)       # (B, 16, 512)      -- 16 patches, 2*16*16=512 patch_dim
    print("mask: ", mask.shape)
    
    patches = patches.squeeze(0) # remove B for now
    mask = mask.squeeze(0) # remove B for now
    re = re.squeeze(0)           # remove B for now

    unrolled = patches.view(64//16, 64//16, C, 16, 16)              # (patch_row, patch_col, C, patch_h, patch_w)
    unrolled = unrolled.permute(2, 0, 3, 1, 4).contiguous()         # (C, patch_row, patch_h, patch_col, patch_w)
    unrolled = unrolled.view(C, 64, 64)                             # (C, grid_size, grid_size)

    unrolled_mask = mask.view(64//16, 64//16, 1, 16, 16)                      # (patch_row, patch_col, C, patch_h, patch_w)
    unrolled_mask = unrolled_mask.permute(2, 0, 3, 1, 4).contiguous()         # (C, patch_row, patch_h, patch_col, patch_w)
    unrolled_mask = unrolled_mask.view(1, 64, 64)                             # (C, grid_size, grid_size)

    u = unrolled[0]                                                 # (grid_size, grid_size)
    v = unrolled[1]                                                 # (grid_size, grid_size)
    P = unrolled[2]                                                 # (grid_size, grid_size)
    m = unrolled_mask[0]

    x_grid = torch.linspace(0, 1, 64)
    y_grid = torch.linspace(0, 1, 64)

    U = np.ma.masked_where(m.numpy(), u.numpy())
    V = np.ma.masked_where(m.numpy(), v.numpy())
    P = np.ma.masked_where(m.numpy(), P.numpy())

    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    a = axes[0].contourf(x_grid, y_grid, U, levels=50, cmap="jet")
    plt.colorbar(a)
    axes[0].set_title("u velocity")

    a = axes[1].contourf(x_grid, y_grid, V, levels=50, cmap="jet")
    axes[1].set_title("v velocity")
    plt.colorbar(a)

    a = axes[2].contourf(x_grid, y_grid, P, levels=50, cmap="jet")
    axes[2].set_title("Pressure")
    plt.colorbar(a)

    a = axes[3].contourf(x_grid, y_grid, m, levels=50, cmap="jet")
    axes[3].set_title("Mask")
    plt.colorbar(a)

    plt.suptitle(f"Re (normalised): {re.item()}")
    plt.tight_layout()
    plt.show()