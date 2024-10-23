import pandas as pd
import torch
import numpy as np


def load_drop_data_from_xlsx(excel_file="data\Qualifying Exam Data.xlsx"):
    """Load in the drop data into a dictionary of tensors"""
    sheets = pd.read_excel(excel_file, sheet_name=None)
    tensors = {}
    for sheet_name, df in sheets.items():
        if sheet_name in ["Temp Profiles", "wt% Profiles"]:
            skip_rows = 3
        else:
            skip_rows = 1
        data_numpy = np.array(
            df.values[
                skip_rows:,
                :,
            ],
            dtype=np.float64,
        )
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        tensors[sheet_name] = data_tensor

    return tensors


if __name__ == "__main__":
    tensors = load_drop_data_from_xlsx()
    for key in tensors:
        print(f"{key}, shape: {tensors[key].shape}")
