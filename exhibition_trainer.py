"""
Interactive OCR Reader for Exhibition Stone Inscriptions
Draw bounding boxes to get character predictions
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import os
from typing import List, Tuple
import glob

class ExhibitionCNN(nn.Module):
    """Same CNN architecture as training"""
    def __init__(self, num_classes):
        super(ExhibitionCNN, self).__init__()
        
        # First block
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        # Second block
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.1)
        
        # Third block
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.1)
        
        # Calculate the size after convolutions
        self.flatten_size = 256 * 6 * 6
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.dropout5 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x

class InteractiveOCRReader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        self.input_size = (48, 48)
        
        # UI state
        self.current_image = None
        self.display_image = None
        self.original_image = None
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.window_name = "Interactive OCR Reader"
        
        # Results
        self.predictions = []
        
        print("Initializing Interactive OCR Reader...")
        self.load_model()
    
    def find_latest_model_files(self):
        """Find the most recent model files"""
        model_files = glob.glob("exhibition_model_*.pth")
        encoder_files = glob.glob("exhibition_encoder_*.pkl")
        metadata_files = glob.glob("exhibition_metadata_*.json")
        
        if not model_files or not encoder_files:
            print("Error: No trained model files found!")
            print("Please make sure you have:")
            print("  - exhibition_model_*.pth")
            print("  - exhibition_encoder_*.pkl")
            print("Run the training script first!")
            return None, None, None
        
        # Get the latest files by timestamp
        model_file = sorted(model_files)[-1]
        encoder_file = sorted(encoder_files)[-1]
        metadata_file = sorted(metadata_files)[-1] if metadata_files else None
        
        return model_file, encoder_file, metadata_file
    
    def load_model(self):
        """Load the trained model"""
        model_file, encoder_file, metadata_file = self.find_latest_model_files()
        
        if model_file is None:
            return False
        
        try:
            # Load label encoder
            with open(encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            num_classes = len(self.label_encoder.classes_)
            print(f"Found {num_classes} classes: {list(self.label_encoder.classes_)}")
            
            # Load model
            self.model = ExhibitionCNN(num_classes).to(self.device)
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.model.eval()
            
            print(f"Model loaded successfully from {model_file}")
            print(f"Encoder loaded from {encoder_file}")
            
            if metadata_file:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"Model trained on {metadata.get('total_annotations', 'unknown')} annotations")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_roi(self, roi, image_path=""):
        """Preprocess ROI exactly like training"""
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Same preprocessing as training
        if "Watage_2" in image_path:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            roi = clahe.apply(roi)
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)
            roi = cv2.fastNlMeansDenoising(roi, None, 10, 7, 21)
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (white on black)
        if np.mean(roi) > 127:
            roi = 255 - roi
        
        # Resize to input size
        roi = cv2.resize(roi, self.input_size, interpolation=cv2.INTER_AREA)
        
        return roi
    
    def predict_character(self, roi, image_path=""):
        """Predict character from ROI"""
        if self.model is None:
            return "?", 0.0
        
        try:
            # Preprocess
            processed_roi = self.preprocess_roi(roi, image_path)
            
            # Convert to tensor
            roi_tensor = torch.FloatTensor(processed_roi).unsqueeze(0).unsqueeze(0) / 255.0
            roi_tensor = roi_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(roi_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            # Decode prediction
            predicted_char = self.label_encoder.classes_[predicted_idx]
            
            return predicted_char, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "?", 0.0
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect = (self.start_point[0], self.start_point[1], x, y)
                self.draw_interface()
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                
                # Get bounding box coordinates
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure positive width and height
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    # Extract ROI from original image
                    roi = self.original_image[y1:y2, x1:x2]
                    
                    # Predict character
                    predicted_char, confidence = self.predict_character(roi, self.image_path)
                    
                    # Store prediction
                    prediction = {
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'character': predicted_char,
                        'confidence': confidence
                    }
                    self.predictions.append(prediction)
                    
                    print(f"Predicted: '{predicted_char}' (confidence: {confidence:.3f})")
                
                self.current_rect = None
                self.draw_interface()
    
    def draw_interface(self):
        """Draw the interface with predictions"""
        display = self.display_image.copy()
        
        # Draw existing predictions
        for pred in self.predictions:
            x, y, w, h = pred['bbox']
            char = pred['character']
            conf = pred['confidence']
            
            # Choose color based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            
            # Draw character and confidence
            label = f"{char} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(display, (x, y-25), (x+label_size[0]+10, y-5), color, -1)
            cv2.putText(display, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw current rectangle being drawn
        if self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Instructions
        instructions = [
            "Draw boxes around characters to predict them",
            "Press 'c' to clear all predictions",
            "Press 'e' to export results",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display, instruction, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, display)
    
    def load_image(self, image_path):
        """Load image for OCR"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            print(f"Error: Cannot load image {image_path}")
            return False
        
        # Create display image (resized if too large)
        self.current_image = self.original_image.copy()
        max_height = 800
        if self.current_image.shape[0] > max_height:
            scale = max_height / self.current_image.shape[0]
            new_width = int(self.current_image.shape[1] * scale)
            self.display_image = cv2.resize(self.current_image, (new_width, max_height))
            # Note: This affects bounding box coordinates, but for simplicity we'll work on display coords
            self.original_image = self.display_image.copy()
        else:
            self.display_image = self.current_image.copy()
        
        self.predictions = []
        
        print(f"Loaded image: {image_path}")
        print(f"Image size: {self.display_image.shape[:2]}")
        
        return True
    
    def export_results(self):
        """Export prediction results"""
        if not self.predictions:
            print("No predictions to export")
            return
        
        # Create text output
        text_output = ""
        for pred in self.predictions:
            text_output += pred['character']
        
        # Save results
        results = {
            'image_path': self.image_path,
            'predictions': self.predictions,
            'recognized_text': text_output,
            'total_characters': len(self.predictions)
        }
        
        output_file = f"ocr_results_{len(self.predictions)}_chars.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults exported to {output_file}")
        print(f"Recognized text: '{text_output}'")
        print(f"Total characters: {len(self.predictions)}")
    
    def run(self, image_path):
        """Run interactive OCR"""
        if self.model is None:
            print("Error: Model not loaded. Cannot proceed.")
            return
        
        if not self.load_image(image_path):
            return
        
        print("\n" + "="*60)
        print("INTERACTIVE OCR READER")
        print("="*60)
        print("\nControls:")
        print("  Mouse: Draw bounding boxes around characters")
        print("  'c': Clear all predictions")
        print("  'e': Export results to JSON")
        print("  'q': Quit")
        print("\nConfidence color coding:")
        print("  Green: High confidence (>80%)")
        print("  Yellow: Medium confidence (50-80%)")
        print("  Red: Low confidence (<50%)")
        print("="*60)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.draw_interface()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.predictions = []
                print("Cleared all predictions")
                self.draw_interface()
            elif key == ord('e'):
                self.export_results()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    reader = InteractiveOCRReader()
    
    # Default images - change these to your image paths
    image_options = [
        r"E:\OCR1\Watage_2.png",
        r"E:\OCR1\Screenshot 2025-09-12 125450.png"
    ]
    
    print("\nAvailable images:")
    for i, img_path in enumerate(image_options):
        if os.path.exists(img_path):
            print(f"  {i+1}: {os.path.basename(img_path)}")
        else:
            print(f"  {i+1}: {os.path.basename(img_path)} (NOT FOUND)")
    
    try:
        choice = input("\nSelect image number (or enter custom path): ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(image_options):
            selected_image = image_options[int(choice)-1]
        else:
            selected_image = choice
        
        if os.path.exists(selected_image):
            reader.run(selected_image)
        else:
            print(f"Error: Image not found: {selected_image}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")