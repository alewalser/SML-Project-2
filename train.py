"""Training model on the ETHMugs dataset."""

import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from eth_mugs_dataset import ETHMugsDataset
from utils import compute_iou, postprocess_mask, TverskyLoss, DiceLoss
from test import run_test_predictions, ensemble_predict
from model import build_model


def train(ckpt_dir: str, train_data_root: str, val_data_root: str, train_indices=None, val_indices=None):
    """
    Train the model on the ETHMugs dataset and evaluate using validation data.
    Args:
        ckpt_dir (str): Directory to save checkpoints.
        train_data_root (str): Path to training images.
        val_data_root (str): Path to validation images.
        train_indices (list): Indices of training samples.
        val_indices (list): Indices of validation samples.
    """
    # Training configuration
    num_epochs = 1
    lr = 1e-3
    lr_patience = 2
    train_batch_size = 8
    val_batch_size = 1
    val_frequency = 1
    epochs_without_improvement = 0
    early_stop_patience = 10
    
    # Create checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load datasets and create data loaders
    train_dataset_og = ETHMugsDataset(train_data_root, mode="train", indices=train_indices)
    train_dataset_hflip = ETHMugsDataset(train_data_root, mode = "hflip", indices = train_indices)
    train_dataset_vflip = ETHMugsDataset(train_data_root, mode = "vflip", indices = train_indices)
    train_dataset = train_dataset_og + train_dataset_hflip + train_dataset_vflip
    val_dataset = ETHMugsDataset(val_data_root, mode="val", indices=val_indices)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # Initialize model, loss, optimizer, and learning rate scheduler
    model = build_model().to(device)
    bce, tversky, dice = torch.nn.BCEWithLogitsLoss(), TverskyLoss(), DiceLoss() # Define loss functions
    criterion = dice # Choose loss function here (bce, tversky, dice)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=lr_patience)

    print("[INFO]: Starting training...")
    best_val_iou = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train phase
        model.train()
        epoch_loss = 0.0

        for image, gt_mask in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
            image, gt_mask = image.to(device), gt_mask.to(device).float()

            optimizer.zero_grad()
            output = model(image)['out']
            loss = criterion(output, gt_mask)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"[INFO] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        # Validation phase
        if epoch % val_frequency == 0:
            model.eval()
            val_iou = 0.0

            with torch.no_grad():
                for i, (val_image, val_gt_mask) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)):
                    val_image, val_gt_mask = val_image.to(device), val_gt_mask.to(device).float()

                    val_output = model(val_image)['out']
                    val_output = torch.sigmoid(val_output)
                    prob_map = val_output.squeeze().cpu().numpy()
                    
                    # Mask Post-Processing
                    pred_mask = postprocess_mask(prob_map)
                    
                    # Ground truth
                    gt_mask = val_gt_mask.squeeze().cpu().numpy().astype("uint8")

                    val_iou += compute_iou(pred_mask, gt_mask)

                val_iou /= len(val_loader)
                print(f"[INFO]: Validation IoU: {val_iou:.2f}")

                # Check if validation improves and apply early stopping otherwise
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
                    print(f"[INFO]: Best model saved at epoch {epoch+1}")
                else:
                    epochs_without_improvement += 1
                    print(f"[INFO]: No improvement for {epochs_without_improvement} epoch(s)")

                if epochs_without_improvement >= early_stop_patience:
                    print(f"[INFO]: Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Adjust learning rate
        lr_scheduler.step(val_iou)

    return model


def kfold_train(data_root, ckpt_dir, num_folds):
    """
    Perform k-fold cross-validation training.
    Args:
        data_root (str): Path to dataset.
        ckpt_dir (str): Where to store model checkpoints.
        num_folds (int): Number of folds.
    """
    # Load full dataset to get total indices
    full_dataset = ETHMugsDataset(data_root, mode="train")
    all_indices = list(range(len(full_dataset)))

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        print(f"[INFO] Starting Fold {fold + 1}/{num_folds}")
        fold_ckpt_dir = os.path.join(ckpt_dir, f"fold_{fold + 1}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        # Train the model for this fold using normal train function
        train(fold_ckpt_dir, data_root, data_root, train_indices=train_idx, val_indices=val_idx)

        model_path = os.path.join(fold_ckpt_dir, "best_model.pth")
        best_model_paths.append(model_path)
        print(f"[INFO] Fold {fold + 1} completed. Best model at {model_path}")
    
    return best_model_paths


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument("--kfold", action="store_true", help="Use K-Fold cross-validation instead of a single train/val split.")
    parser.add_argument("-d", "--data_root", default="./datasets", help="Path to the datasets folder.",)
    parser.add_argument("--ckpt_dir", default="./checkpoints", help="Path to save the model checkpoints to.",)
    parser.add_argument("--predict_test", action="store_true", help="Run test prediction after training")
    parser.add_argument("--test_data_root", type=str, help="Path to test RGB data (e.g., ./datasets/test_data)")
    parser.add_argument("--save_test_dir", type=str, default="final_test_prediction", help="Where to save test predictions")
    args = parser.parse_args()

    # Generate a timestamped directory for this run's checkpoints
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    if args.kfold:
        # Train the model
        model_paths = kfold_train(args.data_root, ckpt_dir, num_folds=2)
        # Use the best models for prediction
        print(f"[INFO]: Using best models from {model_paths} for prediction on test data.")
        ensemble_predict(model_paths, args.test_data_root, save_path_dir=os.path.join(ckpt_dir, "ensemble_prediction"))
        
    else:
        # Set up paths to training and validation data
        full_dataset = ETHMugsDataset(args.data_root, mode="train")
        all_indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
        # Train the model
        model = train(ckpt_dir, args.data_root, args.data_root, train_indices=train_idx, val_indices=val_idx)
        # Use the best model for prediction
        best_model_path = os.path.join(ckpt_dir, "best_model.pth")
        print(f"[INFO]: Using best model from {best_model_path} for prediction on test data.")
        run_test_predictions(model, best_model_path, args.test_data_root, save_path_dir=args.save_test_dir)
    