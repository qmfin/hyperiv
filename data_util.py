import numpy as np
import torch
from torch.utils.data import Dataset


def find_closest_elements(df, col_name, ref_list):
    unique_elements = np.array(df[col_name].unique())
    ref_list = np.array(ref_list)
    closest_elements = []
    for value in ref_list:
        closest_index = np.abs(unique_elements - value).argmin()
        closest_elements.append(unique_elements[closest_index])
        unique_elements = np.delete(unique_elements, closest_index)
    return df[df[col_name].isin(np.array(closest_elements))]


class OptionDataset(Dataset):
    def __init__(self, df, N=1024, sample=True):
        self.full_data = df
        self.few_data = df[df["is_ref"] == 1]
        self.N = N
        self.sample = sample
        self.dates = self.few_data["date"].unique()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]

        few_data_selected = self.few_data[self.few_data["date"] == date]
        z = few_data_selected[["log_moneyness", "tau", "implied_volatility"]].values

        full_data_selected = self.full_data[self.full_data["date"] == date]

        if self.sample:
            sampled_full_data = full_data_selected.sample(n=self.N, replace=True)
        else:
            sampled_full_data = full_data_selected

        X = sampled_full_data[["log_moneyness", "tau"]].values
        y = sampled_full_data["implied_volatility"].values

        return (
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
