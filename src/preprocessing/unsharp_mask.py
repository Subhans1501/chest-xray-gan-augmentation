import cv2
import numpy as np


class UnsharpMaskPreprocessing:
    def __init__(self, kernel_size: tuple = (5, 5), sigma: float = 1.0, amount: float = 1.5, threshold: int = 0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img_np, self.kernel_size, self.sigma)
        mask = img_np.astype(np.float32) - blurred.astype(np.float32)

        if self.threshold > 0:
            mask[np.abs(mask) < self.threshold] = 0

        enhanced = img_np.astype(np.float32) + self.amount * mask
        return np.clip(enhanced, 0, 255).astype(np.uint8)
