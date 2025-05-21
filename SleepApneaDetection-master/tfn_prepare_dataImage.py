import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

def extract_image_features(image_paths, image_size=(224, 224)):
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    features = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = model(img_tensor).squeeze().numpy()
            features.append(feat)
        else:
            features.append(np.zeros(512))  # fallback for missing image

    return np.array(features)

def prepare_data_with_images(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    df['Sex'] = df['Sex'].astype(str).str.upper().replace('M', 'M').replace('F', 'F')
    sex_encoder = LabelEncoder()
    df['Sex'] = sex_encoder.fit_transform(df['Sex'])

    severity_encoder = LabelEncoder()
    df['Severity'] = severity_encoder.fit_transform(df['Severity'])

    image_paths = df['Record'].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))
    image_features = extract_image_features(image_paths)

    clinical_features = df[['Age', 'Sex', 'AHI']].values
    signal_features = df[['Signal_Mean', 'Signal_Std', 'Signal_Max', 'Signal_Min']].values
    target = df['Severity'].values

    return clinical_features, signal_features, image_features, target, sex_encoder, severity_encoder
