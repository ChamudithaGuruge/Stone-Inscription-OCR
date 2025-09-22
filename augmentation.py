"""
Enhanced Training Script for Exhibition OCR using Advanced Augmentation
Updated for E:/OCR2 directory structure
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime
from scipy import ndimage
import random
from collections import defaultdict  # Added missing import

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StoneInscriptionAugmenter:
    """Advanced augmentation specifically for stone inscription characters"""
    
    def __init__(self):
        self.elastic_alpha = 20
        self.elastic_sigma = 3
        
    def elastic_transform(self, image, alpha=None, sigma=None):
        """Elastic deformation to simulate stone wear patterns"""
        if alpha is None:
            alpha = self.elastic_alpha
        if sigma is None:
            sigma = self.elastic_sigma
            
        shape = image.shape
        dx = np.random.uniform(-1, 1, shape) * alpha
        dy = np.random.uniform(-1, 1, shape) * alpha
        
        dx = ndimage.gaussian_filter(dx, sigma, mode='constant', cval=0)
        dy = ndimage.gaussian_filter(dy, sigma, mode='constant', cval=0)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        distorted = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
        return distorted.reshape(shape)
    
    def simulate_erosion(self, image, severity=0.1):
        """Simulate stone erosion patterns"""
        eroded = image.copy()
        
        # Random erosion spots
        h, w = image.shape
        num_spots = int(h * w * severity * 0.01)
        
        for _ in range(num_spots):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(1, 3)
            
            # Create erosion pattern
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            
            # Apply erosion
            eroded = cv2.bitwise_and(eroded, cv2.bitwise_not(mask))
        
        return eroded
    
    def add_surface_texture(self, image, intensity=0.3):
        """Add stone surface texture noise"""
        noise = np.random.normal(0, intensity * 255, image.shape)
        
        # Create texture pattern
        texture = np.sin(np.linspace(0, 4*np.pi, image.shape[1])) * 10
        texture = np.tile(texture, (image.shape[0], 1))
        
        # Combine noise and texture
        combined_noise = noise + texture
        
        # Apply to image
        textured = image.astype(np.float32) + combined_noise
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        
        return textured
    
    def perspective_distortion(self, image, strength=0.1):
        """Simulate viewing angle variations"""
        h, w = image.shape
        
        # Random perspective points
        offset = int(min(h, w) * strength)
        
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [random.randint(-offset, offset), random.randint(-offset, offset)],
            [w + random.randint(-offset, offset), random.randint(-offset, offset)],
            [w + random.randint(-offset, offset), h + random.randint(-offset, offset)],
            [random.randint(-offset, offset), h + random.randint(-offset, offset)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (w, h), borderValue=0)
        
        return warped
    
    def lighting_variation(self, image, shadow_strength=0.3):
        """Simulate uneven lighting conditions"""
        h, w = image.shape
        
        # Create gradient lighting
        light_x = random.randint(w//4, 3*w//4)
        light_y = random.randint(h//4, 3*h//4)
        
        # Create lighting map
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - light_x)**2 + (y - light_y)**2)
        max_distance = np.sqrt(h**2 + w**2)
        
        # Normalize and apply
        lighting = 1.0 - (distance / max_distance) * shadow_strength
        lighting = np.clip(lighting, 0.5, 1.0)
        
        lit_image = (image.astype(np.float32) * lighting).astype(np.uint8)
        
        return lit_image
    
    def augment_character(self, roi, num_augmentations=30):
        """Apply comprehensive augmentation to a character ROI"""
        augmented = [roi]  # Include original
        
        for i in range(num_augmentations):
            aug = roi.copy()
            
            # Basic geometric transformations (light)
            if random.random() > 0.5:
                angle = random.uniform(-5, 5)
                h, w = aug.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                aug = cv2.warpAffine(aug, M, (w, h), borderValue=0)
            
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                h, w = aug.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
                aug = cv2.warpAffine(aug, M, (w, h), borderValue=0)
            
            # Stone-specific augmentations
            if random.random() > 0.7:
                aug = self.elastic_transform(aug)
            
            if random.random() > 0.8:
                aug = self.simulate_erosion(aug, severity=random.uniform(0.05, 0.15))
            
            if random.random() > 0.6:
                aug = self.add_surface_texture(aug, intensity=random.uniform(0.1, 0.4))
            
            if random.random() > 0.7:
                aug = self.perspective_distortion(aug, strength=random.uniform(0.05, 0.15))
            
            if random.random() > 0.5:
                aug = self.lighting_variation(aug, shadow_strength=random.uniform(0.1, 0.4))
            
            # Light noise
            if random.random() > 0.6:
                noise = np.random.normal(0, random.uniform(2, 8), aug.shape)
                aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            augmented.append(aug)
        
        return augmented

class CharacterDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class ExhibitionCNN(nn.Module):
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

class EnhancedExhibitionOCRTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.char_to_bbox_stats = {}
        self.annotations = []
        self.input_size = (48, 48)
        self.device = device
        self.augmenter = StoneInscriptionAugmenter()
        
        # Updated paths for E:/OCR2
        self.data_dir = r"E:\OCR2"
        self.annotations_file = os.path.join(self.data_dir, "exhibition_annotations.json")
        self.images = [
            os.path.join(self.data_dir, "Watage_2.png"),
            os.path.join(self.data_dir, "Screenshot 2025-09-12 125450.png")
        ]
        
    def load_annotations(self):
        """Load exhibition annotations from E:/OCR2"""
        try:
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.annotations_file} not found!")
            print("Please make sure you have created the annotations file first.")
            return False
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return False
        
        print(f"Loaded {len(self.annotations)} annotations from E:\OCR2")
        
        # Update image paths to E:\OCR2
        for ann in self.annotations:
            old_path = ann['image']
            if 'E:\\OCR1\\' in old_path:
                ann['image'] = old_path.replace('E:\\OCR1\\', 'E:\\OCR2\\')
            elif not old_path.startswith('E:\\OCR2\\'):
                # Handle relative paths
                filename = os.path.basename(old_path)
                ann['image'] = os.path.join(self.data_dir, filename)
        
        # Calculate bbox statistics per character
        from collections import defaultdict
        char_bboxes = defaultdict(list)
        
        for ann in self.annotations:
            char = ann['label']
            bbox = ann['bbox']
            char_bboxes[char].append(bbox)
        
        # Store statistics for inference
        for char, bboxes in char_bboxes.items():
            widths = [b[2] for b in bboxes]
            heights = [b[3] for b in bboxes]
            
            self.char_to_bbox_stats[char] = {
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'std_width': np.std(widths),
                'std_height': np.std(heights),
                'count': len(bboxes),
                'aspect_ratios': [w/h for w, h in zip(widths, heights)]
            }
        
        print(f"Found {len(char_bboxes)} unique characters")
        
        # Print character distribution
        sorted_chars = sorted(char_bboxes.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nCharacter distribution:")
        for char, bboxes in sorted_chars[:15]:
            print(f"  '{char}': {len(bboxes)} instances")
        
        return True
    
    def extract_roi(self, image_path: str, bbox: List[int]) -> np.ndarray:
        """Extract character region with enhanced preprocessing"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load {image_path}")
        
        x, y, w, h = bbox
        
        # Add small padding to capture context
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        roi = img[y1:y2, x1:x2]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhanced image-specific preprocessing
        if "Watage_2" in image_path:
            # Enhanced preprocessing for Watage_2.png
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            roi = clahe.apply(roi)
            
            # Adaptive thresholding
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((2,2), np.uint8)
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
            
        else:
            # Enhanced preprocessing for Screenshot and other images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)
            
            # Noise reduction
            roi = cv2.fastNlMeansDenoising(roi, None, 10, 7, 21)
            
            # Adaptive thresholding
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (white on black)
        if np.mean(roi) > 127:
            roi = 255 - roi
        
        # Resize to input size
        roi = cv2.resize(roi, self.input_size, interpolation=cv2.INTER_AREA)
        
        return roi
    
    def show_augmentation_statistics(self):
        """Show detailed augmentation statistics and examples"""
        print("\n" + "="*60)
        print("AUGMENTATION STATISTICS")
        print("="*60)
        
        # Character distribution statistics
        char_counts = defaultdict(int)
        for ann in self.annotations:
            char_counts[ann['label']] += 1
        
        print(f"\nOriginal Dataset:")
        print(f"Total annotations: {len(self.annotations)}")
        print(f"Unique characters: {len(char_counts)}")
        
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nCharacter distribution:")
        for char, count in sorted_chars:
            print(f"  '{char}': {count} samples")
        
        # Augmentation impact
        total_original = len(self.annotations)
        augmentation_factor = 31  # 1 original + 30 augmented
        total_augmented = total_original * augmentation_factor
        
        print(f"\nAfter Augmentation (30x per sample):")
        print(f"Total samples: {total_augmented}")
        print(f"Samples per character: {total_augmented / len(char_counts):.1f}")
        print(f"Data increase: {augmentation_factor}x")
        
        # Show augmentation examples
        self.show_augmentation_examples()
        
        return char_counts
    
    def show_augmentation_examples(self):
        """Show visual examples of augmentation techniques"""
        print(f"\nGenerating augmentation examples...")
        
        # Select a few characters for demonstration
        demo_annotations = []
        seen_chars = set()
        
        for ann in self.annotations:
            if ann['label'] not in seen_chars and len(demo_annotations) < 5:
                demo_annotations.append(ann)
                seen_chars.add(ann['label'])
        
        # Create augmentation examples
        fig, axes = plt.subplots(len(demo_annotations), 8, figsize=(16, 2*len(demo_annotations)))
        if len(demo_annotations) == 1:
            axes = axes.reshape(1, -1)
        
        for i, ann in enumerate(demo_annotations):
            try:
                # Extract original ROI
                original_roi = self.extract_roi(ann['image'], ann['bbox'])
                
                # Generate augmented versions
                augmented_rois = self.augmenter.augment_character(original_roi, num_augmentations=7)
                
                # Display original + 7 augmented versions
                for j, roi in enumerate(augmented_rois[:8]):
                    axes[i, j].imshow(roi, cmap='gray')
                    if j == 0:
                        axes[i, j].set_title(f"'{ann['label']}'\nOriginal", fontsize=10)
                    else:
                        axes[i, j].set_title(f"Aug {j}", fontsize=8)
                    axes[i, j].axis('off')
                
            except Exception as e:
                print(f"Error creating example for '{ann['label']}': {e}")
        
        plt.tight_layout()
        example_path = os.path.join(self.data_dir, 'augmentation_examples.png')
        plt.savefig(example_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Augmentation examples saved to: {example_path}")
    
    def analyze_augmentation_impact(self, sample_roi, char_label):
        """Analyze the impact of different augmentation techniques"""
        print(f"\nAnalyzing augmentation impact for character '{char_label}':")
        
        techniques = {
            'Original': lambda x: x,
            'Elastic': lambda x: self.augmenter.elastic_transform(x),
            'Erosion': lambda x: self.augmenter.simulate_erosion(x, 0.1),
            'Texture': lambda x: self.augmenter.add_surface_texture(x, 0.3),
            'Perspective': lambda x: self.augmenter.perspective_distortion(x, 0.1),
            'Lighting': lambda x: self.augmenter.lighting_variation(x, 0.3)
        }
        
        # Apply each technique
        results = {}
        for name, func in techniques.items():
            try:
                augmented = func(sample_roi.copy())
                
                # Calculate statistics
                stats = {
                    'mean_intensity': np.mean(augmented),
                    'std_intensity': np.std(augmented),
                    'min_intensity': np.min(augmented),
                    'max_intensity': np.max(augmented),
                    'black_pixels': np.sum(augmented == 0),
                    'white_pixels': np.sum(augmented == 255)
                }
                
                results[name] = {
                    'image': augmented,
                    'stats': stats
                }
                
                print(f"  {name:12}: Mean={stats['mean_intensity']:.1f}, "
                      f"Std={stats['std_intensity']:.1f}, "
                      f"Black={stats['black_pixels']}, "
                      f"White={stats['white_pixels']}")
                
            except Exception as e:
                print(f"  {name:12}: Error - {e}")
        
        return results
    
    def prepare_dataset(self):
        """Prepare enhanced training data with advanced augmentation"""
        
        # Show augmentation statistics first
        char_counts = self.show_augmentation_statistics()
        
        X = []
        y = []
        augmentation_stats = defaultdict(int)
        
        print("\nPreparing enhanced dataset with advanced augmentation...")
        
        for i, ann in enumerate(self.annotations):
            try:
                roi = self.extract_roi(ann['image'], ann['bbox'])
                
                # Analyze first character's augmentation impact
                if i == 0:
                    self.analyze_augmentation_impact(roi, ann['label'])
                
                # Use enhanced augmentation
                augmented_rois = self.augmenter.augment_character(roi, num_augmentations=30)
                
                # Track augmentation statistics
                augmentation_stats[ann['label']] += len(augmented_rois)
                
                for aug_roi in augmented_rois:
                    X.append(aug_roi)
                    y.append(ann['label'])
                
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(self.annotations)} annotations")
                    
            except Exception as e:
                print(f"  Error processing annotation {i}: {e}")
        
        if len(X) == 0:
            return None, None
        
        # Show final statistics
        print(f"\nFinal Augmentation Statistics:")
        for char in sorted(augmentation_stats.keys()):
            original_count = char_counts[char]
            augmented_count = augmentation_stats[char]
            print(f"  '{char}': {original_count} → {augmented_count} samples ({augmented_count/original_count:.1f}x)")
        
        # Convert to arrays and normalize
        X = np.array(X)
        X = X.reshape(X.shape[0], 1, self.input_size[0], self.input_size[1])
        X = X.astype('float32') / 255.0
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nEnhanced dataset ready: {X.shape[0]} samples")
        print(f"Classes: {len(self.label_encoder.classes_)}")
        print(f"Average samples per character: {X.shape[0] / len(self.label_encoder.classes_):.1f}")
        
        return X, y_encoded
    
    def train(self):
        """Train model with enhanced augmentation"""
        if not self.load_annotations():
            return None
        
        X, y = self.prepare_dataset()
        if X is None or y is None:
            return None
        
        # Create datasets
        dataset = CharacterDataset(X, y)
        
        # Split dataset
        val_split = 0.05
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) if val_size > 0 else None
        
        print(f"\nTraining samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create model
        num_classes = len(self.label_encoder.classes_)
        self.model = ExhibitionCNN(num_classes).to(self.device)
        
        print(f"\nModel created with {num_classes} classes")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss and optimizer with improved settings
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0003)  # Lower learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_accuracy = 0.0
        patience = 30  # Increased patience
        patience_counter = 0
        
        print("\nStarting enhanced training...")
        
        for epoch in range(300):  # More epochs for better convergence
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_loss /= len(val_loader)
                val_accuracy = val_correct / val_total
                
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                scheduler.step(val_loss)
                current_accuracy = train_accuracy
            else:
                current_accuracy = train_accuracy
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(self.data_dir, 'exhibition_model_enhanced_best.pth'))
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                if val_loader is not None:
                    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}')
                else:
                    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.data_dir, 'exhibition_model_enhanced_best.pth')))
        
        # Plot history
        self.plot_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        # Save model
        self.save_model()
        
        print(f"\nEnhanced training completed! Best accuracy: {best_accuracy:.2%}")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def plot_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(train_accuracies, label='Training')
        if val_accuracies:
            axes[0].plot(val_accuracies, label='Validation')
        axes[0].set_title('Enhanced Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(train_losses, label='Training')
        if val_losses:
            axes[1].plot(val_losses, label='Validation')
        axes[1].set_title('Enhanced Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.data_dir, 'exhibition_enhanced_training_history.png')
        plt.savefig(plot_path, dpi=100)
        plt.show()
        
        print(f"\nTraining history saved to {plot_path}")
    
    def save_model(self):
        """Save enhanced model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model state dict
        model_file = os.path.join(self.data_dir, f'exhibition_enhanced_model_{timestamp}.pth')
        torch.save(self.model.state_dict(), model_file)
        print(f"\nEnhanced model saved to {model_file}")
        
        # Save label encoder
        encoder_file = os.path.join(self.data_dir, f'exhibition_enhanced_encoder_{timestamp}.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Enhanced encoder saved to {encoder_file}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_file': model_file,
            'encoder_file': encoder_file,
            'input_size': self.input_size,
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'char_stats': self.char_to_bbox_stats,
            'total_annotations': len(self.annotations),
            'framework': 'pytorch',
            'torch_version': torch.__version__,
            'device': str(self.device),
            'images': self.images,
            'data_directory': self.data_dir,
            'enhanced_augmentation': True,
            'augmentation_methods': [
                'elastic_transform', 'erosion_simulation', 'surface_texture',
                'perspective_distortion', 'lighting_variation', 'geometric_transforms'
            ]
        }
        
        metadata_file = os.path.join(self.data_dir, f'exhibition_enhanced_metadata_{timestamp}.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Enhanced metadata saved to {metadata_file}")
        
        print(f"\nAll enhanced files saved in E:/OCR2 with timestamp: {timestamp}")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Working directory: E:/OCR2")
    
    trainer = EnhancedExhibitionOCRTrainer()
    result = trainer.train()
    
    if result is None:
        print("\nEnhanced training failed. Please check the error messages above.")
    else:
        print("\nEnhanced training completed successfully!")
        print("Expected improvements:")
        print("- Better accuracy due to advanced augmentation")
        print("- More robust character recognition")
        print("- Better handling of stone inscription artifacts")