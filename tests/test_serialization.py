import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.serialization import serialize_output, deserialize_output
import json

def test_serialization():
    # Test image serialization
    print("Testing image serialization...")
    # Create a sample image tensor (3 channels, 64x64)
    image_tensor = torch.rand(3, 64, 64)
    serialized_image = serialize_output(image_tensor, "IMAGE")
    print("Serialized image:", json.dumps(serialized_image, indent=2))
    
    # Deserialize and verify
    deserialized_image = deserialize_output(serialized_image)
    print("Image shape matches:", image_tensor.shape == deserialized_image.shape)
    print("Image content similar:", torch.allclose(image_tensor, deserialized_image, rtol=1e-4))
    
    # Test latent serialization
    print("\nTesting latent serialization...")
    latent_tensor = torch.randn(1, 4, 32, 32)  # Typical latent shape
    serialized_latent = serialize_output(latent_tensor, "LATENT")
    print("Serialized latent:", json.dumps(serialized_latent, indent=2))
    
    # Deserialize and verify
    deserialized_latent = deserialize_output(serialized_latent)
    print("Latent shape matches:", latent_tensor.shape == deserialized_latent.shape)
    print("Latent content matches:", torch.allclose(latent_tensor, deserialized_latent))
    
    # Test small tensor serialization (uses base64)
    print("\nTesting small tensor serialization...")
    small_tensor = torch.tensor([1.0, 2.0, 3.0])
    serialized_tensor = serialize_output(small_tensor, "TENSOR")
    print("Serialized small tensor:", json.dumps(serialized_tensor, indent=2))
    
    # Deserialize and verify
    deserialized_tensor = deserialize_output(serialized_tensor)
    print("Small tensor content matches:", torch.allclose(small_tensor, deserialized_tensor))
    
    # Test regular JSON-serializable data
    print("\nTesting regular data serialization...")
    regular_data = {"name": "test", "value": 42}
    serialized_regular = serialize_output(regular_data, "OTHER")
    print("Serialized regular data:", json.dumps(serialized_regular, indent=2))
    
    # Clean up test files
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith(("tensor_", "image_")):
                os.remove(os.path.join(output_dir, file))

if __name__ == "__main__":
    test_serialization()
