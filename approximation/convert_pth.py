import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def pth_to_numpy(model_path):
    logger.info(f"Loading model: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    flat_list = []
    metadata = {}
    current_idx = 0
    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()
        flat = arr.flatten()
        n = flat.size
        metadata[key] = {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "start": current_idx,
            "end": current_idx + n
        }
        flat_list.extend(flat)
        current_idx += n
    flat_array = np.array(flat_list)
    logger.info(f"Total parameters: {len(flat_array)}")
    return flat_array, metadata

def numpy_to_pth(flat_array, metadata, reference_model_path, output_path):
    reference_state_dict = torch.load(reference_model_path, map_location="cpu")
    new_state_dict = {}
    for key, info in metadata.items():
        start = info["start"]
        end = info["end"]
        values = flat_array[start:end]
        reshaped = values.reshape(info["shape"])
        tensor = torch.tensor(reshaped, dtype=reference_state_dict[key].dtype)
        new_state_dict[key] = tensor
    torch.save(new_state_dict, output_path)
    logger.info(f"Saved reconstructed model: {output_path}")


def plot_weight_histogram(flat_array, bins=100, log_scale=False):
    plt.figure(figsize=(10,5))
    plt.hist(flat_array, bins=bins, color='blue', alpha=0.7)    
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    if log_scale:
        plt.yscale("log")
    plt.show()