import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import wfdb

def download_ptbxl_if_needed(download_dir="./dataset/ptbxl"):
    zip_path = os.path.join(download_dir, "ptbxl.zip")
    extract_dir = os.path.join(download_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    csv_path = os.path.join(extract_dir, "ptbxl_database.csv")

    if not os.path.exists(csv_path):
        os.makedirs(download_dir, exist_ok=True)
        print("📥 Downloading PTB-XL dataset...")
        url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip"
        urllib.request.urlretrieve(url, zip_path)

        print("📦 Unzipping PTB-XL...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)

        print("✅ PTB-XL dataset is ready.")

class PTBXLDataset(Dataset):
    def __init__(self, df, data_path, leads, transform=None):
        self.df = df
        self.data_path = data_path
        self.leads = leads
        self.transform = transform
        self.mlb = MultiLabelBinarizer()
        self.labels = self.mlb.fit_transform(self.df['scp_codes'].map(eval).map(lambda x: list(x.keys())))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = os.path.join(self.data_path, row['filename_lr'].replace('.dat', ''))
        signals, _ = wfdb.rdsamp(record_path)
        signals = signals.T
        if self.transform:
            signals = self.transform(signals)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(signals, dtype=torch.float32), label

def load_partition(data_dir="./dataset/ptbxl", batch_size=64, validation_split=0.2):
    download_ptbxl_if_needed(data_dir)
    extract_dir = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    df = pd.read_csv(os.path.join(extract_dir, "ptbxl_database.csv"))

    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=validation_split, random_state=42)

    train_dataset = PTBXLDataset(train_df, extract_dir, leads)
    val_dataset = PTBXLDataset(val_df, extract_dir, leads)
    test_dataset = PTBXLDataset(test_df, extract_dir, leads)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    return load_partition(batch_size=batch_size)[1]
