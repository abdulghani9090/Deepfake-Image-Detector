
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import torch.nn.functional as F

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_and_crop(self, image):
        """
        Detects face in PIL Image.
        Returns:
            cropped_face (PIL Image): The cropped face if found.
            found (bool): True if face found, False otherwise.
            original_image (PIL Image): The original image (for fallback).
        """
        # Convert PIL to CV2 (RGB)
        img_np = np.array(image)
        
        # Detect faces
        try:
            faces = self.detector.detect_faces(img_np)
        except Exception as e:
            # Fallback for MTCNN errors
            print(f"Face detection error: {e}")
            faces = []
        
        if not faces:
            # Try a simple center crop fallback if detection fails? 
            # For now just return original
            return image, False, image

        # Get the detection with highest confidence
        # faces is a list of dicts: {'box': [x, y, w, h], 'confidence': float, ...}
        best_face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = best_face['box']
        
        # Add some padding
        padding = 0.2 # Increased padding for better context
        h_pad = int(h * padding)
        w_pad = int(w * padding)
        
        # Clip coordinates
        x1 = max(0, x - w_pad)
        y1 = max(0, y - h_pad)
        x2 = min(img_np.shape[1], x + w + w_pad)
        y2 = min(img_np.shape[0], y + h + h_pad)
        
        cropped_np = img_np[y1:y2, x1:x2]
        cropped_pil = Image.fromarray(cropped_np)
        
        return cropped_pil, True, image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook for gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        # Forward pass
        output = self.model(x)
        
        # Target for backprop (the predicted class score)
        score = output[:, 0] # Assuming single output for binary classification
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weigh activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.data.squeeze().cpu().numpy()

def predict_with_tta(model, input_tensor):
    """
    Test Time Augmentation (TTA) for robust inference.
    Averages predictions across:
    1. Original Image
    2. Horizontal Flip
    """
    model.eval()
    with torch.no_grad():
        # Pass 1: Original
        output_orig = model(input_tensor)
        prob_orig = torch.sigmoid(output_orig).item()
        
        # Pass 2: Horizontal Flip
        input_flip = torch.flip(input_tensor, [3]) # Flip width dimension
        output_flip = model(input_flip)
        prob_flip = torch.sigmoid(output_flip).item()
        
    # Ensemble (Average)
    avg_prob = (prob_orig + prob_flip) / 2.0
    
    return avg_prob

def overlay_heatmap(heatmap, image_pil, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image with thresholding for clarity.
    """
    img_np = np.array(image_pil)
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # 1. Thresholding: Remove noise (low activation areas)
    heatmap[heatmap < 0.2] = 0
    
    # 2. Smoothing: Gaussian Blur for organic looking heatmaps
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # Apply colormap
    heatmap_uint8 = (255 * heatmap).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blending
    # Make regions with no heatmap transparent-ish on the heatmap layer
    mask = (heatmap > 0.05).astype(np.float32)[:, :, np.newaxis]
    
    # Weighted blend only where heatmap is active
    blended = (heatmap_colored * alpha + img_np * (1 - alpha)) * mask + img_np * (1 - mask)
    
    return Image.fromarray(blended.astype(np.uint8))
