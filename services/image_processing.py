"""
Image processing utilities for the Multi-Modal RAG system.
This module provides functions for processing and filtering images, particularly for
detecting and removing blank or uniform images from the dataset.
"""

import base64
import io
from PIL import Image
import numpy as np

def is_blank_image(image: Image.Image, stddev_thresh: float = 10.0) -> bool:
    """
    Check if an image is blank or uniform.
    
    This function determines if an image is blank by analyzing the standard deviation
    of pixel intensities. A low standard deviation indicates a uniform image (blank,
    completely black, or completely white).
    
    Args:
        image (Image.Image): PIL Image object to analyze
        stddev_thresh (float): Threshold for standard deviation (default: 10.0)
            Images with standard deviation below this value are considered blank
            
    Returns:
        bool: True if the image is considered blank, False otherwise
    """
    gray = image.convert("L")  # Convert to grayscale
    np_img = np.array(gray)

    stddev = np.std(np_img)
    return stddev < stddev_thresh

def filter_non_blank_images(base64_images: list[str]) -> list[str]:
    """
    Filter out blank images from a list of base64-encoded images.
    
    This function processes a list of base64-encoded images and removes any that
    are determined to be blank or uniform. It handles potential errors in image
    decoding and processing gracefully.
    
    Args:
        base64_images (list[str]): List of base64-encoded image strings
        
    Returns:
        list[str]: List of base64-encoded strings containing only non-blank images
        
    Note:
        Invalid or corrupted images are skipped and logged with an error message.
    """
    non_blank_images = []
    for b64 in base64_images:
        try:
            image_data = base64.b64decode(b64)
            image = Image.open(io.BytesIO(image_data))
            if not is_blank_image(image):
                non_blank_images.append(b64)
        except Exception as e:
            print(f"Skipping invalid image: {e}")
    return non_blank_images
