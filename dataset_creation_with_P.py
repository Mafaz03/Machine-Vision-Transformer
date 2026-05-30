import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import matplotlib.tri as tri

from scipy.interpolate import griddata
from tqdm import tqdm


def generate_data_from_txt(file_pth: str, save_folder: str):
    master_df = pd.read_csv(
        f"{file_pth}",
        comment="%",
        header=None
    )

    x = np.array(master_df.iloc[:, 0])
    y = np.array(master_df.iloc[:, 1])

    with open(file_pth) as f:
        for line in f:
            if line.startswith("%"):
                re_data = line.strip().split(",")

    Re_s = ([float(i.split("@ ")[-1].split("=")[-1]) for i in re_data[2:]])
    Re_s_unique = []
    for i in range(0, len(Re_s), 3):
        Re_s_unique.append(Re_s[i])

    for re_idx in tqdm(range(0, len(Re_s_unique))):
        start = 2 + 3*re_idx
        sample = master_df.iloc[:, start:start+3]

        u = np.array(sample.iloc[:, 0])
        v = np.array(sample.iloc[:, 1])
        p = np.array(sample.iloc[:, 2])
        
        arr = np.column_stack([x, y, u, v, p])
        df = pd.DataFrame(arr)
        df.columns = ['x', 'y', 'u (m/s)', 'v (m/s)', 'p (Pa)']

        df.to_csv(f"{save_folder}/Re_{Re_s_unique[re_idx]}.csv")

        # points = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis = 1)

        # lin = np.linspace(0, 1, 64)
        # grid_x, grid_y = np.meshgrid(lin, lin)

        # u_grid = griddata(points, u, (grid_x, grid_y), method="linear", fill_value=0.0)
        # v_grid = griddata(points, v, (grid_x, grid_y), method="linear", fill_value=0.0)
        # p_grid = griddata(points, p, (grid_x, grid_y), method="linear", fill_value=0.0)

        # # cbar = plt.contourf(lin, lin, np.sqrt(u_grid**2 + v_grid**2), cmap = "jet", levels = 100)
        # cbar = plt.contourf(lin, lin, p_grid, cmap = "jet", levels = 100)
        # plt.colorbar(cbar)

        # plt.title(f"Re: {Re_s_unique[re_idx]}")    
    
    print("GENERATED!!")

if __name__ == "__main__":
    generate_data_from_txt("simulation_data_with_P.txt", "Data_with_P")