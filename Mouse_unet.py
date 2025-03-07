# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import os
from unet1 import UNet
import logging
from tqdm import tqdm
import numpy as np
import cv2

logging.basicConfig(
    filename="training_log.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger()
logger.addHandler(console_handler)

# mice = ['../../Datasets/optoacousticsparse/mice_sparse16_recon_256.mat', '../../Datasets/optoacousticsparse/mice_sparse32_recon_256.mat', '../../Datasets/optoacousticsparse/mice_sparse128_recon_256.mat']
# v_phantom = ['../../Datasets/optoacousticsparse/phantom_sparse16_recon_256.mat', '../../Datasets/optoacousticsparse/phantom_sparse32_recon_256.mat', '../../Datasets/optoacousticsparse/phantom_full_recon_256.mat']
HDF5_INPUT_PATH = '../../Datasets/optoacousticsparse/mice_sparse16_recon_256.mat'
HDF5_OUTPUT_PATH = '../../Datasets/optoacousticsparse/mice_sparse128_recon_256.mat'
GAMMA = 0.98
MOMENTUM = 0.9
STEP_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 250
LOSS_MUL = 1e4
num_patches = 100
n_samples = 1000
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
BATCH_SIZE = 1
START_FILTERS = 32
MODEL_SAVE_PATH = "unet_model_mouse16_2.pth.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cuda_available = torch.cuda.is_available()
current_device = torch.cuda.current_device() if cuda_available else None
gpu_name = torch.cuda.get_device_name(
    current_device) if cuda_available else "No GPU"
cpu_count = os.cpu_count()

# logger.info(f"Available CPU cores: {cpu_count}")
# logger.info(f"CUDA Available: {cuda_available}")
# logger.info(
#     f"Current CUDA Device: {current_device if cuda_available else 'N/A'}")
# logger.info(f"GPU Name: {gpu_name}")

cpu_count = os.cpu_count()
num_workers = cpu_count / 2
num_workers = int(num_workers)
# logger.info(f"Num workers: {num_workers}")


class DataloaderMouse(Dataset):
    def __init__(self, input_file_path, output_file_path, input_key=None,
                 output_key=None,
                 indices=None, normalize=True):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.normalize = normalize
        with h5py.File(input_file_path, 'r') as infile:
            self.input_key = input_key or list(infile.keys())[0]
            input_size = len(infile[self.input_key])
        self.indices = indices if indices is not None else list(
            range(input_size))
        with h5py.File(input_file_path, 'r') as infile:
            self.input_key = input_key or list(infile.keys())[0]

        with h5py.File(output_file_path, 'r') as outfile:
            self.output_key = output_key or list(outfile.keys())[0]

        if indices is None:
            with h5py.File(input_file_path, 'r') as infile:
                self.indices = list(range(len(infile[self.input_key])))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        with h5py.File(self.input_file_path, 'r') as f_input, h5py.File(
                self.output_file_path, 'r') as f_output:
            x = torch.tensor(f_input[self.input_key][real_idx],
                             dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(f_output[self.output_key][real_idx],
                             dtype=torch.float32).unsqueeze(0)

            if self.normalize:
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                # x = cv2.resize(x.numpy().squeeze(0), (256, 256),
                #                interpolation=cv2.INTER_LINEAR)
                # x = torch.tensor(x).unsqueeze(0)

                y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                # y = cv2.resize(y.numpy().squeeze(0), (256, 256),
                #                interpolation=cv2.INTER_LINEAR)
                # y = torch.tensor(y).unsqueeze(0)

        return x, y
if __name__ == '__main__':
    EARLY_STOPPING_PATIENCE = 50
    best_val_loss = float('inf')
    early_stop_counter = 0
    dataset = DataloaderMouse(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, normalize=True)
    size = len(dataset)
    indices = np.arange(size)

    # train_end = int(0.6 * size) ### this is split for Vphantom!!!
    train_end = int(0.7 * size)
    val_end = train_end + int(0.2 * size)
    # train_indices, val_indices, test_indices = indices[:train_end], indices[train_end:val_end+1], indices[val_end +1:]
    train_indices, val_indices, test_indices = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    logger.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    train_dataset = DataloaderMouse(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, indices=train_indices, normalize=True)
    val_dataset = DataloaderMouse(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, indices=val_indices, normalize=True)
    test_dataset = DataloaderMouse(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, indices=test_indices, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    logger.info(f"Train dataset size: {len(train_dataset)} samples")
    logger.info(f"Validation dataset size: {len(val_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)} samples")

    logging.info(f"Test DataLoader batches: {len(test_loader)}")

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{EPOCHS}...")

        model.train()
        train_loss = 0
        logger.info(f"Starting training for Epoch {epoch + 1}...")
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]") as pbar:
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)
        train_loss /= len(train_loader)
        logger.info(f"Training complete for Epoch {epoch + 1}. Average Training Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0
        logger.info(f"Starting validation for Epoch {epoch + 1}...")
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Validation]") as pbar:
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    y_pred = model(x)
                    loss = F.mse_loss(y_pred, y)
                    val_loss += loss.item()
                    pbar.set_postfix({"Batch Loss": loss.item()})
                    pbar.update(1)
        val_loss /= len(val_loader)
        logger.info(f"Validation complete for Epoch {epoch + 1}. Average Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(),
                       "final_model_weight/Mice/best_model_mouse16_2.pth")
            logger.info(f"Validation loss improved. Model saved as best_model_mouse.pth")
        else:
            early_stop_counter += 1
            logger.info(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered. Stopping training.")
            break

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join("checkpoints", f"Unet_mouse16_2{epoch + 1}.ckpt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Final model saved: {MODEL_SAVE_PATH}")
    # Using my own test in generalevalmice.ipynb
    # logger.info("Starting testing phase...")
    # model.eval()
    # test_loss = 0
    # with tqdm(total=len(test_loader), desc="Testing") as pbar:
    #     with torch.no_grad():
    #         for x, y in test_loader:
    #             x, y = x.to(DEVICE), y.to(DEVICE)
    #             y_pred = model(x)
    #             loss = F.mse_loss(y_pred, y)
    #             test_loss += loss.item()
    #             pbar.update(1)
    # test_loss /= len(test_loader)
    # logger.info(f"Testing complete. Average Test Loss: {test_loss:.4f}")