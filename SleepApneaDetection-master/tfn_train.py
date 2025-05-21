# Imported required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image
import os
from tfn_3model import TFN_3Modal
from tfn_prepare_dataImage import prepare_data_with_images

# Created function for a custom loss that focuses more on classify
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# To load and prepare data
X_clinical, X_signal, X_image, y_all, sex_encoder, severity_encoder = prepare_data_with_images("D:/Dataset/final_multimodal_dataset.csv", image_folder="D:/Dataset/images")

# To normalize features
scaler_clinical = StandardScaler()
X_clinical = scaler_clinical.fit_transform(X_clinical)
scaler_signal = StandardScaler()
X_signal = scaler_signal.fit_transform(X_signal)

# To train/test split
Xc_train, Xc_test, Xs_train, Xs_test, Xi_train, Xi_test, y_train, y_test = train_test_split(
    X_clinical, X_signal, X_image, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# To convert into tensors
Xc_train = torch.tensor(Xc_train, dtype=torch.float32)
Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
Xi_train = torch.tensor(Xi_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

Xc_test = torch.tensor(Xc_test, dtype=torch.float32)
Xs_test = torch.tensor(Xs_test, dtype=torch.float32)
Xi_test = torch.tensor(Xi_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# To set device and create the TFN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TFN_3Modal(clinical_dim=3, signal_dim=4, image_dim=512, fusion_dim=128, num_classes=4).to(device)

Xc_train, Xs_train, Xi_train, y_train = Xc_train.to(device), Xs_train.to(device), Xi_train.to(device), y_train.to(device)
Xc_test, Xs_test, Xi_test, y_test = Xc_test.to(device), Xs_test.to(device), Xi_test.to(device), y_test.to(device)

# To set class weights and handle imbalance to use focal loss to train the model
alpha = torch.tensor([3.5, 2.5, 2.0, 1.0], dtype=torch.float32).to(device)
criterion = FocalLoss(alpha=alpha)

# To optimizer and LR scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# To train model for 100 rounds by making predictions, calculating loss, and updating the model each time
print("\n Training TFN with Images...")
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(Xc_train, Xs_train, Xi_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# To test model without training, check how accurate it is, and show detailed results for each severity level
model.eval()
with torch.no_grad():
    preds = model(Xc_test, Xs_test, Xi_test).argmax(dim=1)
    accuracy = (preds == y_test).sum().item() / y_test.size(0)
    print(f"\n Test Accuracy: {accuracy * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(y_test.cpu(), preds.cpu(), target_names=severity_encoder.classes_, zero_division=1))

# To save predictions
all_preds = model(
    torch.tensor(X_clinical, dtype=torch.float32).to(device),
    torch.tensor(X_signal, dtype=torch.float32).to(device),
    torch.tensor(X_image, dtype=torch.float32).to(device)
).argmax(dim=1).cpu().numpy()

# To decode and save
df_all = pd.read_csv("D:/Dataset/final_multimodal_dataset.csv")
df_all["Predicted_Severity"] = severity_encoder.inverse_transform(all_preds)
if df_all["Sex"].dtype != 'object':
    df_all["Sex"] = sex_encoder.inverse_transform(df_all["Sex"])
if df_all["Severity"].dtype != 'object':
    df_all["Severity"] = severity_encoder.inverse_transform(df_all["Severity"])
df_all["Sex"] = df_all["Sex"].replace("m", "M")
df_all.to_csv("D:/Dataset/patient_predictions.csv", index=False)
print("All-patient predictions saved to patient_predictions.csv")
