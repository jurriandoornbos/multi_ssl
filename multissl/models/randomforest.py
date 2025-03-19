import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, precision_score, recall_score
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from multissl.data import tifffile_loader
from multissl.data.mask_loader import SegmentationDataset
from multissl.plotting.false_color import create_false_color_image, visualize_batch

class RandomForestSegmentation:
    """
    Random Forest based image segmentation model.
    
    This class implements fully-supervised segmentation using scikit-learn's
    RandomForestClassifier. It extracts pixel-based features from multispectral
    images and trains a random forest to predict segmentation masks.
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        pixel_features=True,
        spatial_features=True,
        texture_features=True,
        class_weight='balanced',
        img_size=224,
        in_channels=4
    ):
        """
        Initialize the RandomForestSegmentation model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            n_jobs: Number of jobs to run in parallel
            pixel_features: Whether to use raw pixel values as features
            spatial_features: Whether to include spatial features (x,y coordinates)
            texture_features: Whether to include texture features
            class_weight: Weight for imbalanced classes
            img_size: Size of input images
            in_channels: Number of input channels (e.g., 4 for RGB+NIR)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Feature extraction options
        self.pixel_features = pixel_features
        self.spatial_features = spatial_features  
        self.texture_features = texture_features
        
        # Initialize the classifier
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            verbose=1
        )
        
        # For storing metrics during training
        self.metrics = {
            'train_accuracy': [],
            'val_accuracy': [],
            'val_iou': [],
            'val_f1': []
        }
    
    def extract_features(self, image, include_position=True):
        """
        Extract features from the input image.
        
        Args:
            image: Input image tensor or array of shape [C, H, W] or [H, W, C]
            include_position: Whether to include pixel position features
            
        Returns:
            features: Array of features for each pixel
        """
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] <= 4:  # [C, H, W] format
                image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image
            
        h, w = image_np.shape[0], image_np.shape[1]
        
        # Initialize feature list
        feature_list = []
        
        # Add raw pixel values as features
        if self.pixel_features:
            # Reshape to [H*W, C]
            pixel_values = image_np.reshape(-1, image_np.shape[2])
            feature_list.append(pixel_values)
        
        # Add spatial features (x, y coordinates)
        if self.spatial_features:
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Normalize coordinates to [0, 1]
            x_coords = x_coords / float(w)
            y_coords = y_coords / float(h)
            
            # Reshape to [H*W, 1]
            x_coords = x_coords.reshape(-1, 1)
            y_coords = y_coords.reshape(-1, 1)
            
            feature_list.append(x_coords)
            feature_list.append(y_coords)
        
        # Add texture features if enabled
        if self.texture_features:
            # Calculate simple gradient-based texture features
            if image_np.shape[2] > 1:  # Multi-channel image
                # Use first 3 channels for texture if available
                texture_channels = min(3, image_np.shape[2])
                for i in range(texture_channels):
                    channel = image_np[:, :, i]
                    
                    # Calculate horizontal and vertical gradients
                    grad_x = np.gradient(channel, axis=1).reshape(-1, 1)
                    grad_y = np.gradient(channel, axis=0).reshape(-1, 1)
                    
                    # Magnitude of gradient
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2).reshape(-1, 1)
                    
                    feature_list.append(grad_mag)
            else:  # Single channel
                channel = image_np[:, :, 0]
                grad_x = np.gradient(channel, axis=1).reshape(-1, 1)
                grad_y = np.gradient(channel, axis=0).reshape(-1, 1)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2).reshape(-1, 1)
                feature_list.append(grad_mag)
        
        # Concatenate all features
        features = np.hstack(feature_list)
        
        return features
    
    def fit(self, train_loader, val_loader=None):
        """
        Train the random forest model on the training data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        """
        print("Extracting features from training data...")
        X_train = []
        y_train = []
        
        # Extract features and labels from training data
        for images, masks in tqdm(train_loader):
            # Process each image in the batch
            for i in range(images.shape[0]):
                image = images[i]
                mask = masks[i]
                
                # Extract features
                features = self.extract_features(image)
                
                # Flatten mask to match feature shape
                labels = mask.flatten()
                
                # Add to training data
                X_train.append(features)
                y_train.append(labels)
        
        # Concatenate all training data
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        print(f"Training on {X_train.shape[0]} pixels with {X_train.shape[1]} features per pixel")
        
        # Train the random forest
        print("Training random forest...")
        self.clf.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_preds = self.clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        self.metrics['train_accuracy'].append(train_accuracy)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        # Validate if validation data is provided
        if val_loader is not None:
            self.validate(val_loader)
            
        return self.metrics
    
    def validate(self, val_loader):
        """
        Validate the model on the validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            val_metrics: Dictionary of validation metrics
        """
        print("Validating model...")
        val_accuracies = []
        val_ious = []
        val_f1s = []
        
        # Process each batch in validation data
        for images, masks in tqdm(val_loader):
            batch_size = images.shape[0]
            
            # Process each image in the batch
            for i in range(batch_size):
                image = images[i]
                true_mask = masks[i]
                
                # Predict mask
                pred_mask = self.predict(image)
                
                # Calculate metrics
                accuracy = accuracy_score(true_mask.flatten(), pred_mask.flatten())
                iou = jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='macro')
                f1 = f1_score(true_mask.flatten(), pred_mask.flatten(), average='macro')
                
                val_accuracies.append(accuracy)
                val_ious.append(iou)
                val_f1s.append(f1)
        
        # Calculate mean metrics
        val_accuracy = np.mean(val_accuracies)
        val_iou = np.mean(val_ious)
        val_f1 = np.mean(val_f1s)
        
        # Store metrics
        self.metrics['val_accuracy'].append(val_accuracy)
        self.metrics['val_iou'].append(val_iou)
        self.metrics['val_f1'].append(val_f1)
        
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation IoU: {val_iou:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        
        val_metrics = {
            'accuracy': val_accuracy,
            'iou': val_iou,
            'f1': val_f1
        }
        
        return val_metrics
    
    def predict(self, image):
        """
        Predict segmentation mask for an input image.
        
        Args:
            image: Input image tensor or array
            
        Returns:
            pred_mask: Predicted segmentation mask
        """
        # Extract features
        features = self.extract_features(image)
        
        # Predict
        pred_flat = self.clf.predict(features)
        
        # Reshape to image dimensions
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape[0], image.shape[1]
            
        pred_mask = pred_flat.reshape(h, w)
        
        return pred_mask
    
    def predict_proba(self, image):
        """
        Predict class probabilities for each pixel.
        
        Args:
            image: Input image tensor or array
            
        Returns:
            pred_proba: Predicted class probabilities [H, W, C]
        """
        # Extract features
        features = self.extract_features(image)
        
        # Predict probabilities
        proba_flat = self.clf.predict_proba(features)
        
        # Reshape to image dimensions
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape[0], image.shape[1]
            
        num_classes = proba_flat.shape[1]
        pred_proba = proba_flat.reshape(h, w, num_classes)
        
        return pred_proba
    
    def visualize_predictions(self, images, true_masks=None, num_samples=4):
        """
        Visualize model predictions on sample images.
        
        Args:
            images: Batch of input images
            true_masks: Optional batch of ground truth masks
            num_samples: Number of samples to visualize
            
        Returns:
            fig: Matplotlib figure with visualizations
        """
        # Limit to maximum available samples
        num_samples = min(num_samples, images.shape[0])
        
        if true_masks is not None:
            # Create figure with three rows (images, true masks, predictions)
            fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
            
            for i in range(num_samples):
                # Get image and mask
                image = images[i]
                true_mask = true_masks[i]
                
                # Predict mask
                pred_mask = self.predict(image)
                
                # Create false color image for visualization
                false_color = create_false_color_image(image)
                
                # Plot image
                axes[0, i].imshow(false_color)
                axes[0, i].set_title(f"Image {i}")
                axes[0, i].axis('off')
                
                # Plot ground truth mask
                axes[1, i].imshow(true_mask.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                axes[1, i].set_title(f"Ground Truth {i}")
                axes[1, i].axis('off')
                
                # Plot prediction
                axes[2, i].imshow(pred_mask, cmap='viridis', vmin=0, vmax=1)
                axes[2, i].set_title(f"Prediction {i}")
                axes[2, i].axis('off')
        else:
            # No ground truth, just show images and predictions
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
            
            for i in range(num_samples):
                # Get image
                image = images[i]
                
                # Predict mask
                pred_mask = self.predict(image)
                
                # Create false color image for visualization
                false_color = create_false_color_image(image)
                
                # Plot image
                axes[0, i].imshow(false_color)
                axes[0, i].set_title(f"Image {i}")
                axes[0, i].axis('off')
                
                # Plot prediction
                axes[1, i].imshow(pred_mask, cmap='viridis', vmin=0, vmax=1)
                axes[1, i].set_title(f"Prediction {i}")
                axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath):
        """Save the model to disk"""
        joblib.dump(self.clf, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model from disk"""
        self.clf = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self
        
    def estimate_rf_parameters(self):
        """
        Estimate the number of parameters in a trained Random Forest model
        
        Parameters:
        rf_model: A trained RandomForestClassifier
        n_features: Number of features in the dataset
        
        Returns:
        total_parameters: Estimated number of parameters
        """
        total_parameters = 0
        
        # For each tree in the forest
        for tree in self.clf.estimators_:
            n_nodes = tree.tree_.node_count
            
            # Each non-leaf node stores:
            # - The feature index to split on (1 parameter)
            # - The threshold value for the split (1 parameter)
            
            # Count non-leaf nodes
            n_non_leaf = np.sum(tree.tree_.children_left != -1)
            
            # Parameters for non-leaf nodes: 2 per node (feature index + threshold)
            total_parameters += 2 * n_non_leaf
        
        print("Estimated RF params:", str(total_parameters))
        return total_parameters
