"""Test script for running predictions on ETH Mugs dataset"""

import os
import torch
from PIL import Image
from tqdm import tqdm

import argparse

from eth_mugs_dataset import ETHMugsDataset
from model import build_model
from utils import save_predictions, postprocess_mask


def run_test_predictions(model, checkpoint_path, test_data_root, save_path_dir="prediction"):
    """
    Runs prediction on test images using a trained model and saves the results.
    Args:
        model (torch.nn.Module): The segmentation model.
        checkpoint_path (str): Path to the model weights.
        test_data_root (str): Directory containing test RGB images.
        save_path_dir (str): Directory where results will be saved.
    """
    print(f"[INFO]: Loading the pre-trained model: {checkpoint_path}")

    # Load trained weights and set model to evaluation mode
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    # Load test dataset
    test_dataset = ETHMugsDataset(test_data_root, mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    image_ids = []
    pred_masks = []

    with torch.no_grad(): # Disable gradients for inference to avoid unnecessary computations
        for i, image in enumerate(tqdm(test_loader)):
            image = image.to(device)

            # Model output (probability map)
            output = model(image)['out']
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

            #Post-process mask
            pred_mask = postprocess_mask(prob_map)

            image_ids.append(str(i).zfill(4))  # e.g., "0000"
            pred_masks.append(pred_mask)

            # Save individual predicted mask as an image
            mask_image = Image.fromarray(pred_mask * 255)
            mask_image.save(os.path.join(save_path_dir, f"{image_ids[-1]}_mask.png"))

    # Save CSV for Kaggle submission
    save_predictions(image_ids, pred_masks, save_path=os.path.join(save_path_dir, "submission.csv"))
    print(f"[INFO] Submission saved to {os.path.join(save_path_dir, 'submission.csv')}")


def ensemble_predict(model_paths, test_data_root, save_path_dir="ensemble_prediction"):
    """
    Runs predictions using an ensemble of models and averages their outputs.
    Args:
        model_paths (list): List of paths to model checkpoints.
        test_data_root (str): Path to the test images.
        save_path_dir (str): Where to save the output predictions.
    """
    # Load shared test dataset
    test_dataset = ETHMugsDataset(test_data_root, mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_ids = []
    final_masks = []

    for i, image in enumerate(tqdm(test_loader)):
        image = image.to(device)
        ensemble_prob = None

        # Accumulate predictions from all models
        for model_path in model_paths:
            model = build_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(image)['out']
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                if ensemble_prob is None:
                    ensemble_prob = prob
                else:
                    ensemble_prob += prob

        # Average prediction over models
        ensemble_prob /= len(model_paths)
        pred_mask = (ensemble_prob > 0.5).astype("uint8")

        #Post-process mask
        pred_mask = postprocess_mask(ensemble_prob)

        image_ids.append(str(i).zfill(4))
        final_masks.append(pred_mask)

        # Save image
        os.makedirs(save_path_dir, exist_ok=True)
        mask_img = Image.fromarray(pred_mask * 255)
        mask_img.save(os.path.join(save_path_dir, f"{image_ids[-1]}_mask.png"))

    # Save ensemble results
    save_predictions(image_ids, final_masks, save_path=os.path.join(save_path_dir, "submission.csv"))
    print(f"[INFO]: Ensemble submission saved to {os.path.join(save_path_dir, 'submission.csv')}")


if __name__ == "__main__":
    # Argument parser to run on its own
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./datasets/test_data", help="Path to test data (only rgb/)")
    parser.add_argument("--ckpt", help="Path to the trained model checkpoint")
    parser.add_argument("--save_dir", default="prediction", help="Directory to save predicted masks and CSV")
    parser.add_argument("--ckpts", nargs="+", help="List of model checkpoint paths for ensemble")
    parser.add_argument("--ensemble", action="store_true", help="Enable ensemble prediction")

    args = parser.parse_args()

    if args.ensemble:
        if not args.ckpts:
            raise ValueError("You must provide --ckpts when using --ensemble")
        ensemble_predict(args.ckpts, args.data_root, save_path_dir=args.save_dir)
    else:
        if not args.ckpt:
            raise ValueError("You must provide --ckpt for single-model prediction")
        model = build_model()
        run_test_predictions(model, args.ckpt, args.data_root, save_path_dir=args.save_dir)