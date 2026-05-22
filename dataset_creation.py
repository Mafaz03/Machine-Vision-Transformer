import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import matplotlib.tri as tri

from scipy.interpolate import griddata
from tqdm import tqdm


def generate_data_from_dat(file_pth: str, save_folder: str):
    comment_chars = ("%", "#", "!", "/", ";", "@")
    skip = 0
    lines = []
    with open(file_pth, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and stripped[0] in comment_chars:
                skip += 1
                if stripped and (stripped[2].strip().lower() == "x"):
                    stripped = stripped[2:]
                    print("header found")
                    to_append = re.split(r' {6,}', stripped)
                    lines.append(to_append)

            elif stripped:
                to_append = stripped.split()
                to_append = [float(a) for a in to_append]
                lines.append(to_append)

    if skip:
        print(f"Skipping {skip} header line(s)")

    df = pd.DataFrame(lines)
    df.columns = df.iloc[0] 
    df = df[1:] 


    headers = list(df.columns)
    Re_s = [
        int(x) if float(x).is_integer() else float(x)
        for x in np.unique([float(i.split("=")[-1]) for i in headers if "Re" in i])
    ]

    for idx in tqdm(range(len(Re_s))):
        sub_df = df[["x", "y", f"u (m/s) @ Re={Re_s[idx]}", f"v (m/s) @ Re={Re_s[idx]}"]]
        sub_df = sub_df.rename(columns = {f"u (m/s) @ Re={Re_s[idx]}": f"u (m/s)", f"v (m/s) @ Re={Re_s[idx]}": f"v (m/s)"})
        sub_df.to_csv(f"{save_folder}/Re_{Re_s[idx]}.csv")
    
    print("GENERATED!!")

if __name__ == "__main__":
    generate_data_from_dat("simulation_data.dat", "Data")