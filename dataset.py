import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import pathlib
import cv2

import os

class Image_Dataset(Dataset):

    def __init__(self, root: str = "Data", patch_size: int = 8):
        super().__init__()

        root = pathlib.Path(root)

        self.re_list      = []
        self.text_list    = []
        self.images_list  = []
        self.patches_list = []
        
        files = os.listdir(root)

        for file in files:
            re_value = float(file.split(".jpg")[0])

            image = plt.imread(pathlib.Path.joinpath(root, file))
            image_norm = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_RGB2GRAY)
            image_norm = image_norm.astype(np.float32) / 255.0
            
            image_norm_tensor = torch.tensor(image_norm).unsqueeze(0).unsqueeze(0)  # (1, C, H, W)

            patches = image_norm_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # [1, C, patch_row, patch_col, patch_h, patch_w]
            patches = patches.permute(0, 2, 3, 1, 4, 5)                                                     # [1, patch_row, patch_col, C, patch_h, patch_w]
            _, patch_row, patch_col, C, patch_h, patch_w = patches.shape
            patches = patches.contiguous().view(patch_row * patch_col, C * patch_h * patch_w)               # [num_patches, patch_dim]

            self.images_list.append(image_norm)
            self.patches_list.append(patches)
            self.re_list.append(re_value)
        
        self.re_mean = np.mean(self.re_list)
        self.re_std  = np.std(self.re_list)

    def __len__(self): return len(self.re_list)
    
    def __getitem__(self, index):
        re_value = self.re_list[index]

        # normalize Reynolds number
        re_norm = (re_value - self.re_mean) / self.re_std
        re_tensor = torch.tensor([re_norm], dtype=torch.float32)

        return (re_tensor, self.images_list[index], self.patches_list[index])


if "__main__" == __name__:
    image_dataset = Image_Dataset(root = "Data_Re", patch_size=16)
    dataloader    = DataLoader(image_dataset, batch_size = 1, shuffle = True)

    text, image, patches = next(iter(dataloader))

    print(patches.shape)

    plt.imshow(image[0])
    plt.title(text[0])
    plt.show()