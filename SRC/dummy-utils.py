import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def create_custom_dataloader(data_path, descriptors_path, batch_size, device, drop_last=False):
    """
    Create a custom DataLoader for data and conditions.

    Args:
        data_path (str): Path to the data file.
        descriptors_path (str): Path to the molecular descriptors file.
        batch_size (int): The batch size for the DataLoader.
        device (torch.device): The device to use for tensors.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        DataLoader: A DataLoader instance for your data and conditions.
    """
    # Read data from files
    data = pd.read_csv(data_path, sep="\t")
    descriptors = pd.read_csv(descriptors_path, index_col=0, sep="\t")

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    descriptors = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns, index=descriptors.index)
    data = data.iloc[:, 0:3].join(pd.DataFrame(scaler.fit_transform(data.iloc[:, 3:]), columns=data.columns[3:]))

    S = pd.DataFrame(columns=descriptors.columns)
    M = pd.DataFrame(columns=data.columns[3:])
    T = []
    D = []
    for i in range(len(data)):
        if data.iloc[i].COMPOUND_NAME in descriptors.index:
            S = pd.concat([S, descriptors[descriptors.index == data.iloc[i].COMPOUND_NAME]])
            subset_data = data.iloc[i, 3:].to_frame().T
            M = pd.concat([M, subset_data], ignore_index=True)
            T.append(Time(data.iloc[i].SACRI_PERIOD))
            D.append(Dose(data.iloc[i].DOSE_LEVEL))

    # Convert data to PyTorch tensors
    S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)
    M = torch.tensor(M.to_numpy(dtype=np.float32), device=device)
    T = scaler.fit_transform(np.array(T, dtype=np.float32).reshape(len(T), -1))
    T = torch.tensor(T, device=device)
    D = scaler.fit_transform(np.array(D, dtype=np.float32).reshape(len(D), -1))
    D = torch.tensor(D, device=device)

    # Create a PyTorch Dataset and DataLoader
    dataset = TensorDataset(M, S, T, D)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    return dataloader


def Time(SACRIFICE_PERIOD):
    """
    Map SACRIFICE_PERIOD to normalized time values.
    """
    switcher = {
        '1 day': 1 / 29,
        '4 day': 4 / 29,
        '8 day': 8 / 29,
        '15 day': 15 / 29,
        '29 day': 29 / 29
    }
    return switcher.get(SACRIFICE_PERIOD, 'error')


def Dose(DOSE_LEVEL):
    """
    Map DOSE_LEVEL to normalized dose values.
    """
    switcher = {
        'Low': 0.1,
        'Middle': 0.3,
        'High': 1
    }
    return switcher.get(DOSE_LEVEL, 'error')

