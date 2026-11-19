import numpy as np
import scipy.linalg
import torch
import torch.nn as nn


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-5):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Stabilize covariance matrices before matrix square root
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


def compute_fid(real_loader, generator, device, num_samples=300, latent_dim=100, in_channels=1):
    extractor = FeatureExtractor(in_channels=in_channels).to(device)
    extractor.eval()

    real_features = []
    with torch.no_grad():
        collected = 0
        for imgs, _ in real_loader:
            feats = extractor(imgs.to(device))
            real_features.append(feats.cpu().numpy())
            collected += imgs.size(0)
            if collected >= num_samples:
                break
    real_features = np.concatenate(real_features, axis=0)[:num_samples]

    gen_features = []
    generator.eval()
    with torch.no_grad():
        batch_size = 64
        collected = 0
        while collected < num_samples:
            cur_b = min(batch_size, num_samples - collected)
            z = torch.randn(cur_b, latent_dim, 1, 1, device=device)
            feats = extractor(generator(z))
            gen_features.append(feats.cpu().numpy())
            collected += cur_b
    gen_features = np.concatenate(gen_features, axis=0)[:num_samples]

    mu_r, sigma_r = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


class DownstreamClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
