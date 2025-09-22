"""
Dataset Creator for Exhibition Stone Inscriptions
Specifically for:
- E:\OCR1\Watage_2.png
- E:\OCR1\Screenshot 2025-09-12 125450.png
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class ExhibitionDatasetCreator:
    def __init__(self):
        # Your two specific images
        self.image_paths = [
            r"E:\OCR1\Watage_2.png",
            r"E:\OCR1\Screenshot 2025-09-12 125450.png"
        ]
        
        self.current_image = None
        self.current_image_path = ""
        self.current_idx = 0
        
        self.annotations = []
        self.current_bbox = []
        self.drawing = False
        
        # Common Sinhala characters from your previous work
        self.sinhala_chars = ['ක', 'ත', 'ද', 'න', 'ප', 'ම', 'ය', 'ල', 'ව', 'ස', 'හ', 
                              '්', 'ා', 'ැ', 'ි', 'ු', 'ෙ', 'ො', 'ේ', 'ෝ', 'ෞ',
                              'ං', 'ඃ', 'අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ',
                              'ඍ', 'එ', 'ඒ', 'ඔ', 'ඕ', 'ඛ', 'ග', 'ඝ', 'ච', 'ඡ', 
                              'ජ', 'ඣ', 'ඤ', 'ට', 'ඨ', 'ඩ', 'ණ', 'ථ', 'ධ', 'ඵ', 
                              'බ', 'භ', 'ර', 'ශ', 'ෂ', 'ළ', 'ෆ']
        
        self.window_name = "Exhibition Dataset Creator"
        
        # Load any existing annotations
        self.load_existing_annotations()
    
    def load_existing_annotations(self):
        """Load existing annotations if available"""
        if os.path.exists('exhibition_annotations.json'):
            try:
                with open('exhibition_annotations.json', 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                print(f"Loaded {len(self.annotations)} existing annotations")
                self.print_statistics()
            except:
                print("Starting fresh - no existing annotations")
    
    def print_statistics(self):
        """Print annotation statistics"""
        if not self.annotations:
            return
        
        stats = {}
        for ann in self.annotations:
            img_name = os.path.basename(ann['image'])
            char = ann['label']
            
            if img_name not in stats:
                stats[img_name] = {}
            
            stats[img_name][char] = stats[img_name].get(char, 0) + 1
        
        print("\n=== Annotation Statistics ===")
        for img_name, char_counts in stats.items():
            print(f"\n{img_name}:")
            print(f"  Total: {sum(char_counts.values())} characters")
            print(f"  Unique: {len(char_counts)} different characters")
            # Show top 5 most frequent
            sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top 5: ", end="")
            for char, count in sorted_chars:
                print(f"'{char}':{count} ", end="")
            print()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bboxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_bbox = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_bbox[2] = x
                self.current_bbox[3] = y
                self.draw_current_state()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.current_bbox) == 4:
                x1 = min(self.current_bbox[0], self.current_bbox[2])
                y1 = min(self.current_bbox[1], self.current_bbox[3])
                x2 = max(self.current_bbox[0], self.current_bbox[2])
                y2 = max(self.current_bbox[1], self.current_bbox[3])
                
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.show_character_dialog(x1, y1, x2-x1, y2-y1)
            
            self.current_bbox = []
            self.draw_current_state()
    
    def draw_current_state(self):
        """Draw image with annotations"""
        display = self.current_image.copy()
        
        # Draw existing annotations for current image
        for ann in self.annotations:
            if ann['image'] == self.current_image_path:
                x, y, w, h = ann['bbox']
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Show character label
                cv2.putText(display, ann['label'], (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw current bbox being drawn
        if len(self.current_bbox) == 4 and self.drawing:
            cv2.rectangle(display,
                         (self.current_bbox[0], self.current_bbox[1]),
                         (self.current_bbox[2], self.current_bbox[3]),
                         (255, 0, 0), 2)
        
        cv2.imshow(self.window_name, display)
    
    def show_character_dialog(self, x, y, w, h):
        """Show dialog to select character"""
        root = tk.Tk()
        root.title("Select Sinhala Character")
        root.geometry("500x600")
        
        # Extract and show ROI
        roi = self.current_image[y:y+h, x:x+w]
        roi_display = cv2.resize(roi, (min(200, w*4), min(200, h*4)), 
                                interpolation=cv2.INTER_NEAREST)
        roi_rgb = cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(roi_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        label_img = tk.Label(root, image=imgtk)
        label_img.pack(pady=10)
        
        # Create scrollable frame for characters
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(frame, height=300)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        selected_char = tk.StringVar()
        
        # Add character buttons in grid
        for i, char in enumerate(self.sinhala_chars):
            btn = tk.Radiobutton(scrollable_frame, text=char, value=char,
                                variable=selected_char, font=("Arial", 20))
            btn.grid(row=i//8, column=i%8, padx=3, pady=3)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Custom entry
        tk.Label(root, text="Or type custom character:").pack()
        custom_entry = tk.Entry(root, font=("Arial", 16))
        custom_entry.pack(pady=5)
        
        def save_annotation():
            char = selected_char.get() or custom_entry.get()
            if char:
                annotation = {
                    'image': self.current_image_path,
                    'bbox': [x, y, w, h],
                    'label': char,
                    'timestamp': datetime.now().isoformat()
                }
                self.annotations.append(annotation)
                
                # Save immediately
                self.save_annotations()
                
                print(f"Added '{char}' at position [{x},{y},{w},{h}]")
                root.destroy()
                self.draw_current_state()
            else:
                messagebox.showwarning("Warning", "Please select a character")
        
        def skip():
            root.destroy()
        
        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Save (Enter)", command=save_annotation, 
                 bg="green", fg="white", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Skip (Esc)", command=skip, 
                 bg="red", fg="white", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        
        # Bind keys
        root.bind('<Return>', lambda e: save_annotation())
        root.bind('<Escape>', lambda e: skip())
        
        root.mainloop()
    
    def preprocess_image(self, img):
        """Enhance image for better visibility"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for display
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    
    def load_image(self, idx):
        """Load image by index"""
        self.current_idx = idx
        self.current_image_path = self.image_paths[idx]
        
        img = cv2.imread(self.current_image_path)
        if img is None:
            print(f"Error loading {self.current_image_path}")
            return False
        
        # Apply preprocessing
        self.current_image = self.preprocess_image(img)
        
        # Scale if too large
        max_height = 800
        if self.current_image.shape[0] > max_height:
            scale = max_height / self.current_image.shape[0]
            new_width = int(self.current_image.shape[1] * scale)
            self.current_image = cv2.resize(self.current_image, (new_width, max_height))
            
            # Scale existing annotations
            for ann in self.annotations:
                if ann['image'] == self.current_image_path and 'scaled' not in ann:
                    ann['bbox'] = [int(b * scale) for b in ann['bbox']]
                    ann['scaled'] = True
        
        return True
    
    def save_annotations(self):
        """Save all annotations"""
        with open('exhibition_annotations.json', 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.annotations)} annotations to exhibition_annotations.json")
    
    def run(self):
        """Main annotation loop"""
        print("\n" + "="*60)
        print("EXHIBITION DATASET CREATOR")
        print("="*60)
        print("\nControls:")
        print("  Mouse: Draw bounding boxes around characters")
        print("  'n': Next image")
        print("  'p': Previous image")
        print("  'u': Undo last annotation")
        print("  's': Show statistics")
        print("  'q': Save and quit")
        print("\nTips:")
        print("  - Be precise with bounding boxes")
        print("  - Include vowel marks with consonants")
        print("  - Annotate even partially visible characters")
        print("="*60)
        
        # Start with first image
        if not self.load_image(0):
            return
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.draw_current_state()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.save_annotations()
                self.print_statistics()
                cv2.destroyAllWindows()
                break
            
            elif key == ord('n'):
                # Next image
                if self.current_idx < len(self.image_paths) - 1:
                    self.load_image(self.current_idx + 1)
                    self.draw_current_state()
                else:
                    print("Already at last image")
            
            elif key == ord('p'):
                # Previous image
                if self.current_idx > 0:
                    self.load_image(self.current_idx - 1)
                    self.draw_current_state()
                else:
                    print("Already at first image")
            
            elif key == ord('u'):
                # Undo last annotation for current image
                for i in range(len(self.annotations)-1, -1, -1):
                    if self.annotations[i]['image'] == self.current_image_path:
                        removed = self.annotations.pop(i)
                        print(f"Removed '{removed['label']}'")
                        self.save_annotations()
                        self.draw_current_state()
                        break
            
            elif key == ord('s'):
                self.print_statistics()

if __name__ == "__main__":
    creator = ExhibitionDatasetCreator()
    creator.run()