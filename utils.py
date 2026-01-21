import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
import logging
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def setup_logging(log_file='training.log'):
    """Configure logging to output simultaneously to both a file and the console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        print(f"Unable to open the log file {log_file}: {e}", file=sys.stderr)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def set_no_requires_grad(model):
    for p in model.parameters():
        p.requires_grad = False

def calculate_metrics(y_true, y_pred):
    """Calculate and return the SRCC, LCC, and MSE"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Filter out NaN or Inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {"SRCC": 0.0, "LCC": 0.0, "MSE": np.inf}

    srcc, _ = scipy.stats.spearmanr(y_true, y_pred)
    lcc, _ = scipy.stats.pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return {
        "SRCC": srcc if not np.isnan(srcc) else 0.0,
        "LCC": lcc if not np.isnan(lcc) else 0.0,
        "MSE": mse
    }

def count_parameters(model):
    """Compute and print the total number of parameters and the number of trainable parameters for the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameter count: ")
    logging.info(f"  - Total parameters: {total_params / 1e6:.2f} M")
    logging.info(f"  - Trainable parameters: {trainable_params / 1e6:.2f} M")

def plot_scatterplot(predictions, labels, metrics, output_path):
    """Plot and save a scatter plot of the predicted values against the actual values."""
    predictions_np = np.array(predictions)
    labels_np = np.array(labels)

    plt.figure(figsize=(8, 6))
    plt.scatter(labels_np, predictions_np, alpha=0.5, color='blue', marker='o', label='Predictions')
    
    # Retrieve values from the metrics dictionary
    lcc = metrics.get('LCC', 0.0)
    mse = metrics.get('MSE', 0.0)
    
    plt.title(f'PCC: {lcc:.3f}, MSE: {mse:.3f}', fontsize=16)
    plt.xlabel('Ground Truth MOS', fontsize=14)
    plt.ylabel('Predicted MOS', fontsize=14)
    plt.grid(True)
    
    min_val = min(labels_np.min(), predictions_np.min()) - 0.1
    max_val = max(labels_np.max(), predictions_np.max()) + 0.1
    limits = [min_val, max_val]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.plot(limits, limits, 'r--', alpha=0.75, zorder=0, label='y=x')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close() 
    logging.info(f"The scatter plot has been saved to: {output_path}")

def encode_and_stack_layers(tokenizer, x):
        # Obtain the initial latent representation via the encoder
        z = tokenizer.encoder(x)  
        
        all_z_q = []
        all_codes = []
        all_commitment_loss = []
        latents = []

        residual = z
        for quantizer in tokenizer.quantizer.quantizers:
            z_q_i, _, _, indices_i, z_e_i = quantizer(residual)

            e = quantizer.in_proj(residual)
            q, _ = quantizer.decode_latents(e)

            commitment_loss_i = F.mse_loss(e.detach(), q.detach(), reduction="none")

            residual = residual - z_q_i

            all_z_q.append(z_q_i)
            all_codes.append(indices_i)
            all_commitment_loss.append(commitment_loss_i)
            latents.append(z_e_i)

        z_q = torch.stack(all_z_q, dim=1)
        codes = torch.stack(all_codes, dim=1)
        commitment_loss = torch.cat(all_commitment_loss, dim=1)
        latents = torch.cat(latents, dim=1)
        

        return z_q, codes, latents, commitment_loss
