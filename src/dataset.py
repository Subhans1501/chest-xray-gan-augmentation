import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.preprocessing.clahe import CLAHEPreprocessing
from src.preprocessing.unsharp_mask import UnsharpMaskPreprocessing


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir: str = "data", img_size: int = 64, channels: int = 1,
                 mode: str = "baseline", sample_count: int = None):
        self.img_size = img_size
        self.channels = channels
        self.mode = mode.lower()
        self.clahe_filter = CLAHEPreprocessing(clip_limit=2.0, tile_grid_size=(8, 8))
        self.unsharp_filter = UnsharpMaskPreprocessing(kernel_size=(5, 5), sigma=1.0, amount=1.5)

        self.image_paths = []
        self.labels = []

        if root_dir and os.path.exists(root_dir):
            for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
                class_folders = glob.glob(os.path.join(root_dir, "**", class_name), recursive=True)
                for folder in class_folders:
                    for ext in ("*.jpeg", "*.jpg", "*.png"):
                        for f in glob.glob(os.path.join(folder, ext)):
                            self.image_paths.append(f)
                            self.labels.append(label)

            # Fallback if images are stored without class subfolders
            if len(self.image_paths) == 0:
                for ext in ("*.jpeg", "*.jpg", "*.png"):
                    for f in glob.glob(os.path.join(root_dir, "**", ext), recursive=True):
                        self.image_paths.append(f)
                        self.labels.append(1 if "pneumonia" in f.lower() else 0)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No chest X-ray images found in '{os.path.abspath(root_dir)}'. "
                "Place NORMAL and PNEUMONIA folders inside the data directory."
            )

        if sample_count and sample_count < len(self.image_paths):
            self.image_paths = self.image_paths[:sample_count]
            self.labels = self.labels[:sample_count]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        if self.channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.array(Image.open(path).convert("L"))
        else:
            img = cv2.imread(path)
            if img is None:
                img = np.array(Image.open(path).convert("RGB"))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.mode == "clahe":
            img = self.clahe_filter(img)
        elif self.mode == "unsharp":
            img = self.unsharp_filter(img)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.transpose(img, (2, 0, 1))

        # Scale to [-1, 1] to match DCGAN Tanh output range
        tensor_img = (torch.from_numpy(img).float() / 127.5) - 1.0
        return tensor_img, self.labels[idx]
