# PTBXL FedOps Silo Client

This repository implements a **FedOps Silo client** for the **PTB-XL ECG dataset**, enabling decentralized federated training using time-series ECG signals. It auto-downloads the dataset and runs cleanly in both local and K8s environments.

---

## 📦 Project Overview

- **Dataset**: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) – 12-lead ECG signals (10s each)
- **Model**: CNN + GRU-based classifier (ConvRNN)
- **Training**: Federated Learning via FedOps
- **Platform**: K8s or local client setup
- **Data**: Automatically downloaded from PhysioNet

---

## 🚀 Quick Start

### Step 1: Clone the Repo
```bash
git clone https://github.com/akeelahamed571/ptbxl-fedops-clean.git
cd ptbxl-fedops-clean
```

### Step 2: Set Up Environment
```bash
conda create -n akeel_multimodal python=3.9 -y
conda activate akeel_multimodal
pip install -r requirements.txt
```

---

## 📂 Directory Structure

```
ptbxl-fedops-clean/
├── client_main.py               # Entry point for FL Silo Client
├── data_preparation.py         # Downloads & loads PTB-XL dataset
├── models.py                   # CNN+GRU model and training utils
├── conf/
│   └── config.yaml             # Task-specific FL config
├── requirements.txt
└── .gitignore
```

---

## 🧠 How Dataset Handling Works

When the client runs:
1. `data_preparation.py` checks for `./dataset/ptbxl`
2. If not found, it downloads and unzips PTB-XL automatically
3. Uses `ptbxl_database.csv` and waveform files for training

> No need to manually download or unzip anything.

---

## ⚙️ How to Run on FedOps

1. Submit this GitHub repo in the **FedOps task creation UI**
2. Match your `config.yaml` → `task_id` with the UI
3. From the root of this repo, run:
```bash
python client_main.py
```

Your Silo will now join the FL task and start training locally.

---

## ✨ Notes

- Tested for Python 3.9+
- Make sure the pod/container has internet to auto-download PTB-XL
- Data is not committed to Git (see `.gitignore`)

---

## 📬 Contact

Maintained by [@akeelahamed571](https://github.com/akeelahamed571)
