import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm


class ECGClassifier(nn.Module):
    def __init__(self, input_channels=12, output_size=5, hidden_dim=128):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, x):
        # x: (batch_size, channels=12, time)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # GRU expects (batch, seq_len, features)
        _, h_n = self.gru(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)  # Concatenate both directions
        out = self.fc(h_n)
        return out


def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
        for epoch in range(epochs):
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")
        return model
    return custom_train_torch


def test_torch():
    def custom_test_torch(model, test_loader, cfg):
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss, all_preds, all_labels = 0, [], []

        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)
                loss = criterion(output, y)
                total_loss += loss.item()
                preds = torch.sigmoid(output) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(test_loader), f1, {"f1_score": f1}
    return custom_test_torch

