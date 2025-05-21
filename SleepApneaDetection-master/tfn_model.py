import torch
import torch.nn as nn

class TFN(nn.Module):
    def __init__(self, clinical_dim, signal_dim, fusion_dim=64, num_classes=4):
        super(TFN, self).__init__()

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


        self.signal_net = nn.Sequential(
            nn.Linear(signal_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(32 + 64, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, clinical_input, signal_input):
        clinical_out = self.clinical_net(clinical_input)
        signal_out = self.signal_net(signal_input)
        fused = torch.cat((clinical_out, signal_out), dim=1)
        return self.fusion_net(fused)
