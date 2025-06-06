"""Utility functions."""

import numpy as np
import pandas as pd

from scipy.ndimage import binary_closing, gaussian_filter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard image size to which all images and masks are resized
IMAGE_SIZE = (252, 378)

def load_mask(mask_path):
    """Loads the segmentation mask from the specified path.
    Args:
        mask_path (str): Path to the mask file. Expects masks named like 'XXXX_mask.png'.
    Returns:
        mask (np.array): A binary mask with values 0 (background) or 1 (ETH mug).
    """
    mask = np.asarray(Image.open(mask_path)).astype(int)
    
    # If the mask is stored in 0–255, convert to binary by dividing
    if mask.max() > 1:
        mask = mask // 255
    return mask


def mask_to_rle(mask):
    """
    Convert a binary mask (2D numpy array) to RLE (column-major).
    Returns a string of space-separated values.
    """
    pixels = mask.flatten(order='F')  # Fortran order (column-major)
    pixels = np.concatenate([[0], pixels, [0]])  # pad with zeros to catch transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]  # calculate run lengths
    return ' '.join(str(x) for x in runs)


def compute_iou(pred_mask, gt_mask, eps=1e-6):
    """Computes the IoU between two numpy arrays: pred_mask and gt_mask.
    Args:
        pred_mask (np.array): dtype:int, shape:(image_height, image_width), values are 0 or 1.
        gt_mask (np.array): dtype:int, shape:(image_height, image_width), values are 0 or 1.
        eps (float): epsilon to smooth the division in order to avoid 0/0.
    Returns:
        iou_score (float)
    """
    intersection = ((pred_mask & gt_mask).astype(float).sum())  # will be zero if gt=0 or pred=0
    union = (pred_mask | gt_mask).astype(float).sum()  # will be zero if both are 0
    iou = (intersection + eps) / (union + eps)  # we smooth our division by epsilon to avoid 0/0
    iou_score = iou.mean()
    return iou_score


def save_predictions(image_ids, pred_masks, save_path='submission.csv'):
    """
    Save predictions in the Kaggle submission format (CSV with RLE encoding).
    Args:
        image_ids (list): List of image identifiers, e.g., ["0000", "0001", ...].
        pred_masks (list): List of 2D binary numpy arrays (predicted masks).
        save_path (str): Path to save the resulting CSV.
    """
    assert len(image_ids) == len(pred_masks)
    
    predictions = {'ImageId': [], 'EncodedPixels': []}

    for i in range(len(image_ids)):
        mask = pred_masks[i]

        # Fallback: If mask is empty, encode as a blank string
        if mask.sum() == 0:
            mask_rle = " "
        else:
            mask_rle = mask_to_rle(mask)

        predictions['ImageId'].append(image_ids[i])
        predictions['EncodedPixels'].append(mask_rle)

    # Create DataFrame and write to CSV
    pd.DataFrame(predictions).to_csv(save_path, index=False)


def sobel_edge_map(tensor):
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
    kernel = torch.stack([sobel_x, sobel_y])  # shape (2, 1, 3, 3)
    
    kernel = kernel.to(tensor.device)
    edge = F.conv2d(tensor.unsqueeze(0), kernel, padding=1)
    edge = torch.norm(edge.squeeze(0), dim=0, keepdim=True)
    return edge


def postprocess_mask(prob_map):
    """
    Apply smoothing and morphological operations to convert a probability map into a clean binary mask.
    """
    # Smooth probability map with Gaussian blur
    blurred = gaussian_filter(prob_map, sigma=1)
    # Sharpen by thresholding
    mask = (blurred > 0.5).astype("uint8")
    # Morphological operations
    mask = binary_closing(mask, structure=np.ones((3, 3))).astype("uint8")
    return mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        return 1 - (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)