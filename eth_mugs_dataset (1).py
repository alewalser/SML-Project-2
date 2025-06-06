"""ETH Mugs Dataset."""
import numpy as np

import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import IMAGE_SIZE, load_mask, sobel_edge_map


class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train", indices=None):
        """
        This dataset class loads the ETH Mugs dataset.
        It will return the resized image according to the scale and mask tensors
        in the original resolution.
        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): One of "train", "val", or "test", indicating the purpose.
            indices (list): List of indices to subset the dataset.
        """
        self.mode = mode
        self.root_dir = root_dir

        # Define paths to RGB images and masks
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks") if mode != "test" else None

        # Load all image paths (sorted to align with masks)
        self.all_image_paths = sorted([
            os.path.join(self.rgb_dir, fname)
            for fname in os.listdir(self.rgb_dir)
            if fname.endswith(".png") or fname.endswith(".jpg")
        ])

        # Load all mask paths (not in test mode)
        self.all_mask_paths = sorted([
            os.path.join(self.mask_dir, fname)
            for fname in os.listdir(self.mask_dir)
            if fname.endswith("_mask.png")
        ]) if self.mask_dir else []

        # Apply subset via indices (for training/validation splits)
        if indices is not None:
            self.image_paths = [self.all_image_paths[i] for i in indices]
            self.mask_paths = [self.all_mask_paths[i] for i in indices] if self.mask_dir else []
        else:
            self.image_paths = self.all_image_paths
            self.mask_paths = self.all_mask_paths

        # Define image transformations for training and validation/test modes
        if self.mode == "hflip":    # Horizontal flip augmentation
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE), 
                transforms.RandomHorizontalFlip(p=1),                 # Random horizontal flip for augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE), 
                transforms.RandomHorizontalFlip(p=1),                 # Random horizontal flip for augmentation
            ])
        elif self.mode == "vflip":  # Vertical flip augmentation
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE), 
                transforms.RandomVerticalFlip(p=1),                 # Random vertical flip for augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE), 
                transforms.RandomVerticalFlip(p=1),                 # Random vertical flip for augmentation
            ])
        else: # Default case for training, validation, or test
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),                      # Resize for consistency
                transforms.ToTensor(),                              # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])

        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in the ETHMugsDataset:", len(self.image_paths))

    def __len__(self):
        #Return the number of samples in the dataset.
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Resize image and mask to IMAGE_SIZE
        image = TF.resize(image, IMAGE_SIZE, antialias=True)
        
        # Apply image transform
        if self.transform is not None:
            image = self.transform(image)
            
        # Add edge channel just like in train mode
        # Compute edge map from grayscale version
        gray = TF.rgb_to_grayscale(image)
        edge_tensor = sobel_edge_map(gray)
        edge_tensor = (edge_tensor - edge_tensor.min()) / (edge_tensor.max() - edge_tensor.min() + 1e-6)
        edge_tensor = 0.3 * edge_tensor

        # Concatenate edge to image: shape becomes (4, H, W)
        image = torch.cat([image, edge_tensor], dim=0)  # (4, H, W)

        if self.mode != "test":
            mask_path = self.mask_paths[idx]
            mask = load_mask(mask_path).astype(np.uint8)
            
            mask = Image.fromarray(mask).resize(IMAGE_SIZE)
            
            mask = TF.resize(mask, IMAGE_SIZE, antialias=True)

            # Apply mask transform
            if self.transform is not None and self.mode == "hflip"or self.mode == "vflip":
                #image = self.transform(image)
                mask = self.mask_transform(mask)
                #mask = torch.float().unsqueeze(0)
            
            # Convert mask to float tensor and add channel
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
            
            return image, mask
        

        return image


