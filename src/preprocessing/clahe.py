import cv2
import numpy as np


class CLAHEPreprocessing:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if img_np.ndim == 2:
            return self.clahe.apply(img_np)
        if img_np.ndim == 3:
            if img_np.shape[2] == 1:
                return np.expand_dims(self.clahe.apply(img_np[:, :, 0]), axis=2)
            if img_np.shape[2] == 3:
                # Apply CLAHE only on the luminance channel to preserve color balance
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = self.clahe.apply(l)
                return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return img_np
