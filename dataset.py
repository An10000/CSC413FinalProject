# dataset.py
import os
from glob import glob

import numpy as np
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class LandscapeColorizationDataset(Dataset):
    """
    Dataset for landscape colorization in CIELab space.
    We use:
        - Input:  L channel  (1, H, W), normalized to [-1, 1]
        - Target: ab channels (2, H, W), normalized to [-1, 1]
    """

    def __init__(self, color_dir, image_size=150):
        """
        Args:
            color_dir (str): directory with RGB color images (.jpg / .png).
                             We assume Kaggle dataset images are here.
            image_size (int): resize shorter side to this size (square).
        """
        self.color_paths = sorted(
            glob(os.path.join(color_dir, '*.jpg')) +
            glob(os.path.join(color_dir, '*.png'))
        )
        if len(self.color_paths) == 0:
            raise ValueError(f"No images found in {color_dir}")

        self.size = image_size
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((image_size, image_size))

    def __len__(self):
        return len(self.color_paths)

    def _rgb_to_lab(self, rgb_img):
        """
        rgb_img: numpy array in [0,1], shape (H, W, 3)
        return: L_norm (1, H, W), ab_norm (2, H, W), both in [-1, 1]
        """
        lab = color.rgb2lab(rgb_img).astype(np.float32)  # L in [0,100], a,b in [-128,127]
        L = lab[:, :, 0:1]      # (H, W, 1)
        ab = lab[:, :, 1:3]     # (H, W, 2)

        # Normalize to roughly [-1, 1]
        L_norm = (L / 50.0) - 1.0        # [0,100] -> [-1,1]
        ab_norm = ab / 128.0             # [-128,127] -> about [-1,1]

        # HWC -> CHW
        L_norm = np.transpose(L_norm, (2, 0, 1))   # (1, H, W)
        ab_norm = np.transpose(ab_norm, (2, 0, 1)) # (2, H, W)

        return L_norm, ab_norm

    def __getitem__(self, idx):
        path = self.color_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.resize(img)
        img_tensor = self.to_tensor(img)  # (3, H, W), [0,1]

        # To numpy HWC for skimage.rgb2lab
        rgb_np = img_tensor.permute(1, 2, 0).numpy()
        L, ab = self._rgb_to_lab(rgb_np)

        L = torch.from_numpy(L)    # (1, H, W)
        ab = torch.from_numpy(ab)  # (2, H, W)

        return L, ab


def create_dataloaders(color_dir, image_size=150,
                       batch_size=16, num_workers=2,
                       val_split=0.1, test_split=0.1,
                       seed=42):
    """
    Create train / val / test DataLoaders from a single directory of images.
    """
    full_dataset = LandscapeColorizationDataset(color_dir, image_size=image_size)
    n = len(full_dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True)

    return {
        "train": make_loader(train_ds, True),
        "val": make_loader(val_ds, False),
        "test": make_loader(test_ds, False),
    }
