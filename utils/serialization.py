import os
import torch
import numpy as np
from PIL import Image
import json
import base64
import io

def get_output_dir():
    # Create an output directory for storing serialized data
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_tensor_to_disk(tensor, filename=None):
    """Save a tensor to disk and return the path"""
    output_dir = get_output_dir()
    if filename is None:
        filename = f"tensor_{hash(tensor.cpu().numpy().tobytes())}.pt"
    filepath = os.path.join(output_dir, filename)
    torch.save(tensor, filepath)
    return filepath

def load_tensor_from_disk(filepath):
    """Load a tensor from disk"""
    return torch.load(filepath)

def save_image_to_disk(image_tensor, filename=None):
    """Save an image tensor to disk and return the path"""
    output_dir = get_output_dir()
    if filename is None:
        filename = f"image_{hash(image_tensor.cpu().numpy().tobytes())}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Convert tensor to PIL Image and save
    if isinstance(image_tensor, torch.Tensor):
        image_np = (255. * image_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
        if len(image_np.shape) == 3:
            if image_np.shape[0] in [1, 3, 4]:  # Channels first
                image_np = np.transpose(image_np, (1, 2, 0))
            if image_np.shape[-1] == 1:  # Grayscale
                image_np = image_np[..., 0]
    else:
        image_np = image_tensor
        
    img = Image.fromarray(image_np)
    img.save(filepath)
    return filepath

def load_image_from_disk(filepath):
    """Load an image from disk"""
    img = Image.open(filepath)
    img_np = np.array(img)
    if len(img_np.shape) == 2:  # Grayscale
        img_np = img_np[..., None]
    if img_np.shape[-1] == 4:  # RGBA
        img_np = img_np[..., :3]  # Convert to RGB
    img_np = np.transpose(img_np, (2, 0, 1))  # Convert to channels first
    return torch.from_numpy(img_np).float() / 255.0

def serialize_output(output, output_type):
    """Serialize different types of outputs to JSON-compatible format"""
    if output_type == "IMAGE":
        image_path = save_image_to_disk(output)
        return {
            "type": "IMAGE",
            "path": image_path,
            "metadata": {
                "shape": list(output.shape) if hasattr(output, 'shape') else None,
                "dtype": str(output.dtype) if hasattr(output, 'dtype') else None
            }
        }
    elif output_type == "LATENT":
        tensor_path = save_tensor_to_disk(output)
        return {
            "type": "LATENT",
            "path": tensor_path,
            "metadata": {
                "shape": list(output.shape),
                "dtype": str(output.dtype)
            }
        }
    elif output_type == "TENSOR":
        # For small tensors, we can encode them directly in base64
        if output.numel() < 1000:  # Only for small tensors
            buffer = io.BytesIO()
            torch.save(output, buffer)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return {
                "type": "TENSOR",
                "data": encoded,
                "metadata": {
                    "shape": list(output.shape),
                    "dtype": str(output.dtype)
                }
            }
        else:
            # For larger tensors, save to disk
            return serialize_output(output, "LATENT")
    else:
        # For JSON serializable data, return as is
        try:
            json.dumps(output)
            return output
        except (TypeError, OverflowError):
            # If not JSON serializable, convert to string representation
            return str(output)

def deserialize_output(serialized_data):
    """Deserialize data back to its original format"""
    if isinstance(serialized_data, dict) and "type" in serialized_data:
        if serialized_data["type"] == "IMAGE":
            return load_image_from_disk(serialized_data["path"])
        elif serialized_data["type"] == "LATENT":
            return load_tensor_from_disk(serialized_data["path"])
        elif serialized_data["type"] == "TENSOR":
            # Decode base64 tensor
            encoded_data = serialized_data["data"]
            buffer = io.BytesIO(base64.b64decode(encoded_data))
            return torch.load(buffer)
    return serialized_data
