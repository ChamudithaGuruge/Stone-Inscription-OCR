"""
GUI Interface for Sinhala OCR Reader
Professional interface for stone inscription character recognition
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
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
from PIL import Image, ImageTk
import threading

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

class ImageCanvas(tk.Canvas):
    """Custom canvas for image display and drawing"""
    
    def __init__(self, parent, width=800, height=600):
        super().__init__(parent, width=width, height=height, bg='gray')
        
        self.image = None
        self.photo_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        
        # Predictions
        self.predictions = []
        
        # Bind events
        self.bind("<Button-1>", self.start_draw)
        self.bind("<B1-Motion>", self.draw_motion)
        self.bind("<ButtonRelease-1>", self.end_draw)
        self.bind("<MouseWheel>", self.zoom)
        
        # Callback for when rectangle is drawn
        self.draw_callback = None
    
    def load_image(self, image_path):
        """Load and display image"""
        try:
            # Load with OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                return False
            
            # Convert to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.original_image = cv_image
            
            # Calculate scale to fit canvas
            canvas_width = self.winfo_width() or 800
            canvas_height = self.winfo_height() or 600
            
            img_height, img_width = cv_image.shape[:2]
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            self.scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            # Resize image
            new_width = int(img_width * self.scale)
            new_height = int(img_height * self.scale)
            
            pil_image = Image.fromarray(cv_image)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.photo_image = ImageTk.PhotoImage(pil_image)
            
            # Center image
            self.offset_x = (canvas_width - new_width) // 2
            self.offset_y = (canvas_height - new_height) // 2
            
            self.display_image()
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def display_image(self):
        """Display image on canvas"""
        self.delete("all")
        
        if self.photo_image:
            self.create_image(self.offset_x, self.offset_y, 
                            anchor=tk.NW, image=self.photo_image)
        
        # Draw predictions
        self.draw_predictions()
    
    def draw_predictions(self):
        """Draw prediction boxes"""
        for i, pred in enumerate(self.predictions):
            x, y, w, h = pred['bbox']
            char = pred['character']
            conf = pred['confidence']
            
            # Scale coordinates
            x = int(x * self.scale) + self.offset_x
            y = int(y * self.scale) + self.offset_y
            w = int(w * self.scale)
            h = int(h * self.scale)
            
            # Choose color based on confidence
            if conf > 0.8:
                color = 'green'
            elif conf > 0.5:
                color = 'orange'
            else:
                color = 'red'
            
            # Draw rectangle
            rect_id = self.create_rectangle(x, y, x+w, y+h, 
                                          outline=color, width=2, 
                                          tags=f"pred_{i}")
            
            # Draw label
            label = f"{char} ({conf:.2f})"
            text_id = self.create_text(x+5, y-10, text=label, 
                                     fill=color, anchor=tk.SW,
                                     font=('Arial', 10, 'bold'),
                                     tags=f"pred_{i}")
    
    def start_draw(self, event):
        """Start drawing rectangle"""
        if self.photo_image:
            self.drawing = True
            self.start_x = event.x
            self.start_y = event.y
    
    def draw_motion(self, event):
        """Draw rectangle while dragging"""
        if self.drawing:
            if self.current_rect:
                self.delete(self.current_rect)
            
            self.current_rect = self.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='blue', width=2, tags="current")
    
    def end_draw(self, event):
        """End drawing and process rectangle"""
        if self.drawing:
            self.drawing = False
            
            if self.current_rect:
                self.delete(self.current_rect)
                self.current_rect = None
            
            # Calculate rectangle in image coordinates
            x1 = min(self.start_x, event.x) - self.offset_x
            y1 = min(self.start_y, event.y) - self.offset_y
            x2 = max(self.start_x, event.x) - self.offset_x
            y2 = max(self.start_y, event.y) - self.offset_y
            
            # Scale back to original image coordinates
            x1 = int(x1 / self.scale)
            y1 = int(y1 / self.scale)
            x2 = int(x2 / self.scale)
            y2 = int(y2 / self.scale)
            
            # Ensure valid rectangle
            if x2 - x1 > 10 and y2 - y1 > 10:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                if self.draw_callback:
                    self.draw_callback(bbox)
    
    def zoom(self, event):
        """Zoom functionality"""
        if self.photo_image:
            factor = 1.1 if event.delta > 0 else 0.9
            self.scale *= factor
            # Redisplay with new scale (simplified - full implementation would be more complex)
    
    def clear_predictions(self):
        """Clear all predictions"""
        self.predictions = []
        self.display_image()
    
    def add_prediction(self, bbox, character, confidence):
        """Add a prediction"""
        prediction = {
            'bbox': bbox,
            'character': character,
            'confidence': confidence
        }
        self.predictions.append(prediction)
        self.display_image()

class SinhalaOCRGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sinhala OCR Reader - Stone Inscriptions")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # OCR Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        self.input_size = (48, 48)
        self.current_image_path = ""
        
        # GUI Components
        self.setup_gui()
        self.setup_menu()
        
        # Load model
        self.load_model()
        
        # Status
        self.update_status("Ready. Load an image to begin OCR.")
    
    def setup_gui(self):
        """Setup GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for image
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Image canvas
        canvas_frame = ttk.LabelFrame(left_frame, text="Image")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.image_canvas = ImageCanvas(canvas_frame, width=800, height=600)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_canvas.draw_callback = self.process_selection
        
        # Controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Load Image", 
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Clear All", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=(0, 5))
        
        # Confidence threshold
        ttk.Label(controls_frame, text="Min Confidence:").pack(side=tk.LEFT, padx=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(controls_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, length=100)
        confidence_scale.pack(side=tk.LEFT, padx=(0, 5))
        self.confidence_label = ttk.Label(controls_frame, text="0.50")
        self.confidence_label.pack(side=tk.LEFT)
        
        # Update confidence label
        def update_confidence_label(*args):
            self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
        self.confidence_var.trace('w', update_confidence_label)
        
        # Right panel
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Model info
        model_frame = ttk.LabelFrame(right_frame, text="Model Information")
        model_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.model_info = tk.Text(model_frame, height=4, wrap=tk.WORD)
        self.model_info.pack(fill=tk.X, padx=5, pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Results list
        self.results_tree = ttk.Treeview(results_frame, columns=('Character', 'Confidence'), 
                                       show='tree headings', height=10)
        self.results_tree.heading('#0', text='#')
        self.results_tree.heading('Character', text='Character')
        self.results_tree.heading('Confidence', text='Confidence')
        
        self.results_tree.column('#0', width=30)
        self.results_tree.column('Character', width=80)
        self.results_tree.column('Confidence', width=80)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Recognized text
        text_frame = ttk.LabelFrame(right_frame, text="Recognized Text")
        text_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.recognized_text = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self.recognized_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image...", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Find latest model files
            model_files = glob.glob("exhibition_model_*.pth")
            encoder_files = glob.glob("exhibition_encoder_*.pkl")
            
            if not model_files or not encoder_files:
                self.update_status("No trained model found. Please train the model first.")
                self.model_info.insert(tk.END, "No trained model found.\nPlease run the training script first.")
                return False
            
            model_file = sorted(model_files)[-1]
            encoder_file = sorted(encoder_files)[-1]
            
            # Load label encoder
            with open(encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load model
            num_classes = len(self.label_encoder.classes_)
            self.model = ExhibitionCNN(num_classes).to(self.device)
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.model.eval()
            
            # Update UI
            model_info = f"Model: {os.path.basename(model_file)}\n"
            model_info += f"Classes: {num_classes}\n"
            model_info += f"Device: {self.device}\n"
            model_info += f"Characters: {', '.join(self.label_encoder.classes_[:10])}"
            if len(self.label_encoder.classes_) > 10:
                model_info += "..."
            
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(tk.END, model_info)
            
            self.update_status(f"Model loaded: {num_classes} character classes")
            return True
            
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            self.update_status(error_msg)
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(tk.END, error_msg)
            return False
    
    def load_image(self):
        """Load image for OCR"""
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image for OCR",
            filetypes=file_types
        )
        
        if filename:
            if self.image_canvas.load_image(filename):
                self.current_image_path = filename
                self.clear_results()
                self.update_status(f"Loaded: {os.path.basename(filename)}")
            else:
                messagebox.showerror("Error", "Failed to load image")
    
    def process_selection(self, bbox):
        """Process selected region"""
        if not self.model or not self.current_image_path:
            return
        
        try:
            # Extract ROI from original image
            x, y, w, h = bbox
            original_image = cv2.imread(self.current_image_path)
            roi = original_image[y:y+h, x:x+w]
            
            # Predict character
            character, confidence = self.predict_character(roi)
            
            # Check confidence threshold
            min_confidence = self.confidence_var.get()
            if confidence >= min_confidence:
                # Add to canvas
                self.image_canvas.add_prediction(bbox, character, confidence)
                
                # Add to results
                item_id = len(self.image_canvas.predictions)
                self.results_tree.insert('', 'end', text=str(item_id),
                                       values=(character, f"{confidence:.3f}"))
                
                # Update recognized text
                self.update_recognized_text()
                
                self.update_status(f"Predicted: '{character}' (confidence: {confidence:.3f})")
            else:
                self.update_status(f"Low confidence prediction rejected: '{character}' ({confidence:.3f})")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def predict_character(self, roi):
        """Predict character from ROI"""
        try:
            # Preprocess (same as training)
            processed_roi = self.preprocess_roi(roi)
            
            # Convert to tensor
            roi_tensor = torch.FloatTensor(processed_roi).unsqueeze(0).unsqueeze(0) / 255.0
            roi_tensor = roi_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(roi_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            predicted_char = self.label_encoder.classes_[predicted_idx]
            return predicted_char, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "?", 0.0
    
    def preprocess_roi(self, roi):
        """Preprocess ROI exactly like training"""
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply same preprocessing as training
        if "Watage_2" in self.current_image_path:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            roi = clahe.apply(roi)
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)
            roi = cv2.fastNlMeansDenoising(roi, None, 10, 7, 21)
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed
        if np.mean(roi) > 127:
            roi = 255 - roi
        
        # Resize
        roi = cv2.resize(roi, self.input_size, interpolation=cv2.INTER_AREA)
        return roi
    
    def clear_all(self):
        """Clear all predictions"""
        self.image_canvas.clear_predictions()
        self.clear_results()
        self.update_status("Cleared all predictions")
    
    def clear_results(self):
        """Clear results panel"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.recognized_text.delete(1.0, tk.END)
    
    def update_recognized_text(self):
        """Update recognized text display"""
        text = ''.join([pred['character'] for pred in self.image_canvas.predictions])
        self.recognized_text.delete(1.0, tk.END)
        self.recognized_text.insert(tk.END, text)
    
    def export_results(self):
        """Export results to file"""
        if not self.image_canvas.predictions:
            messagebox.showwarning("Warning", "No predictions to export")
            return
        
        file_types = [
            ('JSON files', '*.json'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            filetypes=file_types,
            defaultextension='.json'
        )
        
        if filename:
            try:
                recognized_text = ''.join([pred['character'] for pred in self.image_canvas.predictions])
                
                results = {
                    'image_path': self.current_image_path,
                    'predictions': self.image_canvas.predictions,
                    'recognized_text': recognized_text,
                    'total_characters': len(self.image_canvas.predictions),
                    'model_info': {
                        'device': str(self.device),
                        'num_classes': len(self.label_encoder.classes_) if self.label_encoder else 0
                    }
                }
                
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Image: {self.current_image_path}\n")
                        f.write(f"Recognized Text: {recognized_text}\n")
                        f.write(f"Total Characters: {len(self.image_canvas.predictions)}\n\n")
                        f.write("Character Details:\n")
                        for i, pred in enumerate(self.image_canvas.predictions):
                            f.write(f"{i+1}. '{pred['character']}' (confidence: {pred['confidence']:.3f})\n")
                
                self.update_status(f"Results exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Results exported to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
    
    def show_instructions(self):
        """Show instructions dialog"""
        instructions = """
Sinhala OCR Reader Instructions:

1. Load an Image:
   - Click "Load Image" or use File > Load Image
   - Select your stone inscription image

2. Select Characters:
   - Click and drag on the image to draw rectangles around characters
   - The system will automatically predict the character
   - Predictions appear with colored boxes:
     • Green: High confidence (>80%)
     • Orange: Medium confidence (50-80%)
     • Red: Low confidence (<50%)

3. Adjust Settings:
   - Use the "Min Confidence" slider to filter predictions
   - Lower values show more predictions (including uncertain ones)
   - Higher values show only confident predictions

4. Export Results:
   - Click "Export Results" to save predictions
   - Choose JSON format for detailed data
   - Choose TXT format for simple text output

5. Tips:
   - Draw tight boxes around individual characters
   - Include vowel marks with their base characters
   - Be precise with bounding boxes for better accuracy
        """
        
        messagebox.showinfo("Instructions", instructions)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Sinhala OCR Reader v1.0

Developed for stone inscription character recognition
using deep learning techniques.

Features:
• Interactive character selection
• Real-time predictions
• Confidence-based filtering
• Multiple export formats

Built with PyTorch and tkinter
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = SinhalaOCRGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")