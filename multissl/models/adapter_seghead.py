# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from multissl.models.seghead import SegmentationModel, HighResolutionHead


class DomainAdapter(nn.Module):
    """
    Domain adapter module that learns to adapt features from one domain to another.
    Implemented as a residual bottleneck module for efficient adaptation.
    Uses GroupNorm instead of BatchNorm for better stability with small datasets.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_dim = max(in_channels // reduction, 16)  # Ensure minimum channel count
        num_groups = min(32, hidden_dim)  # Ensure we don't have more groups than channels
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(x + self.bottleneck(x))  # Residual connection

class AdapterModule(nn.Module):
    """
    Adapter module that contains all domain adapters for different feature levels.
    This is a separate module with its own loss function.
    """
    def __init__(self, feature_dimensions):
        super().__init__()
        
        self.adapters = nn.ModuleDict({
            'layer1': DomainAdapter(feature_dimensions['layer1']),
            'layer2': DomainAdapter(feature_dimensions['layer2']),
            'layer3': DomainAdapter(feature_dimensions['layer3']),
            'layer4': DomainAdapter(feature_dimensions['layer4'])
        })
    
    def forward(self, features):
        """
        Apply domain adaptation to all feature levels
        """
        adapted_features = {}
        for layer_name, feature in features.items():
            if layer_name in self.adapters:
                adapted_features[layer_name] = self.adapters[layer_name](feature)
            else:
                adapted_features[layer_name] = feature
        
        return adapted_features
    
class DomainAdaptiveSegmentationModel(pl.LightningModule):

    """
    Domain adaptive model with clear separation between components:
    1. Feature extractor (backbone)
    2. Adapter module (with its own loss)
    3. Segmentation head (with its own loss)
    
    The adapters are always used in the segmentation head's forward pass.
    """
    def __init__(
        self, 
        backbone_type="resnet50",
        pretrained_path=None,
        in_channels=4,
        num_classes=2, 
        img_size=224,
        supervised_lr=3e-4,
        adapter_lr=1e-4,
        backbone_lr=5e-5,
        weight_decay=3e-5,
        consistency_weight=0.5,
        consistency_rampup=5,
        min_confidence_threshold=0.2,  # Starting threshold (lower)
        max_confidence_threshold=0.8,  # Target threshold (higher)
        confidence_rampup=20,          # Epochs to ramp up confidence threshold
        class_weights=[1.0, 2.0],
        ce_weight=0.6,
        dice_weight=0.4
    ):
        super().__init__()
        
        # Save parameters
        self.supervised_lr = supervised_lr
        self.adapter_lr = adapter_lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self.min_confidence_threshold = min_confidence_threshold
        self.max_confidence_threshold = max_confidence_threshold
        self.confidence_rampup = confidence_rampup
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Initialize feature extractor (backbone)
        self.create_backbone(backbone_type, pretrained_path, in_channels)
        
        # Detect feature dimensions
        self.feature_dimensions = self._detect_feature_dimensions(in_channels, img_size)
        
        # Create adapter module (separate component with its own loss)
        self.adapter_module = AdapterModule(self.feature_dimensions)
        
        # Create segmentation head
        self.segmentation_head = HighResolutionHead(
            layer1_channels=self.feature_dimensions['layer1'],
            layer2_channels=self.feature_dimensions['layer2'],
            layer3_channels=self.feature_dimensions['layer3'],
            layer4_channels=self.feature_dimensions['layer4'],
            num_classes=num_classes,
            img_size=img_size
        )
        
        # For tracking training
        self._current_epoch = 0
        self.supervised_loss_ema = None
        self.consistency_loss_ema = None
        self.ema_alpha = 0.95
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['feature_extractor'])
    
    def create_backbone(self, backbone_type, pretrained_path, in_channels):
        """Create the feature extractor backbone"""
        # We'll reuse the feature extractor part from SegmentationModel
        temp_model = SegmentationModel(
            backbone_type=backbone_type,
            pretrained_path=pretrained_path,
            in_channels=in_channels,
            num_classes=2  # Doesn't matter, we only use the feature extractor
        )
        
        # Extract just the feature extractor
        self.feature_extractor = temp_model.feature_extractor
    
    def _detect_feature_dimensions(self, in_channels, img_size):
        """Detect feature dimensions from the backbone"""
        # Create dummy input
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(dummy_input)
            
            dimensions = {
                'layer1': features['layer1'].size(1),
                'layer2': features['layer2'].size(1),
                'layer3': features['layer3'].size(1),
                'layer4': features['layer4'].size(1)
            }
            
        return dimensions
    
    def forward(self, x):
        """Full forward pass: backbone → adapters → segmentation head"""
        # Extract features from backbone
        features = self.feature_extractor(x)
        
        # Apply domain adapters
        adapted_features = self.adapter_module(features)
        
        # Apply segmentation head to adapted features
        return self.segmentation_head(adapted_features)
    
    def get_raw_backbone_output(self, x):
        """Get features directly from backbone (for debugging)"""
        return self.feature_extractor(x)
    
    def get_consistency_weight(self):
        """Get consistency weight with ramp-up"""
        if self.consistency_rampup > 0:
            return self.consistency_weight * min(1.0, self._current_epoch / self.consistency_rampup)
        return self.consistency_weight
        
    def get_current_confidence_threshold(self):
        """
        Get the current confidence threshold based on training progress.
        Gradually increases from min to max threshold over the rampup period.
        """
        if self.confidence_rampup > 0:
            # Calculate the ramp-up factor (0 to 1)
            rampup_factor = min(1.0, self._current_epoch / self.confidence_rampup)
            
            # Interpolate between min and max threshold
            current_threshold = self.min_confidence_threshold + \
                               (self.max_confidence_threshold - self.min_confidence_threshold) * rampup_factor
                               
            return current_threshold
        
        # If no rampup, use max threshold
        return self.max_confidence_threshold
    
    def dice_loss(self, outputs, targets, smooth=1.0, weight=None):
        """Dice loss calculation"""
        num_classes = outputs.size(1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Calculate Dice loss
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        
        # Apply class weights if provided
        if weight is not None:
            weighted_dice = dice_score * weight.to(outputs.device)
            return 1.0 - weighted_dice.mean()
        else:
            return 1.0 - dice_score.mean()
    
    def supervised_loss(self, outputs, targets):
        """Supervised loss function (for segmentation head)"""
        # Define class weights for CE loss
        cw = self.class_weights.to(outputs.device)
        
        # Calculate CE loss
        ce_loss = F.cross_entropy(outputs, targets, weight=cw)
        
        # Calculate Dice loss
        dice = self.dice_loss(outputs, targets, weight=cw)
        
        # Combine losses with custom weights
        return self.ce_weight * ce_loss + self.dice_weight * dice
    
    def consistency_loss(self, student_logits, teacher_logits, confidence_mask=None):
        """Consistency loss function (for adapter module)"""
        # Get pseudo-labels from teacher logits
        with torch.no_grad():
            pseudo_labels = torch.argmax(teacher_logits, dim=1)
        
        # Calculate cross-entropy loss
        pixel_loss = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
        
        # Apply confidence mask if provided
        if confidence_mask is not None and confidence_mask.sum() > 0:
            loss = (pixel_loss * confidence_mask.float()).sum() / confidence_mask.sum()
        else:
            loss = pixel_loss.mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step with separate losses for adapters and segmentation head
        """
        labeled_batch, unlabeled_batch = batch
        total_loss = 0.0
        
        # ---------- SUPERVISED TRAINING (SEGMENTATION HEAD) ----------
        images_l, masks_l = labeled_batch
        if images_l.size(0) > 0:
            # Forward pass
            logits_l = self(images_l)
            
            # Calculate supervised loss (for segmentation head)
            seg_loss = self.supervised_loss(logits_l, masks_l)
            
            # Add to total loss
            total_loss += seg_loss
            
            # Update EMA and log
            if self.supervised_loss_ema is None:
                self.supervised_loss_ema = seg_loss.detach().item()
            else:
                self.supervised_loss_ema = self.ema_alpha * self.supervised_loss_ema + \
                                         (1 - self.ema_alpha) * seg_loss.detach().item()
            
            # Log metrics
            self.log("train_supervised_loss", seg_loss)
            self.log("train_supervised_loss_ema", self.supervised_loss_ema)
        
        # ---------- DOMAIN ADAPTATION TRAINING (ADAPTER MODULE) ----------
        if unlabeled_batch.size(0) > 0:
            images_u = unlabeled_batch
            
            # Get teacher predictions (without adaptation) - detached for stability
            with torch.no_grad():
                features = self.feature_extractor(images_u)
                teacher_logits = self.segmentation_head(features)
                
                # Calculate confidence scores
                probs = F.softmax(teacher_logits, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                
                # Get current confidence threshold based on training progress
                current_threshold = self.get_current_confidence_threshold()
                
                # Create confidence mask using current threshold
                confidence_mask = (confidence > current_threshold)
                
                # Log percentage of confident pixels and current threshold
                confident_percent = confidence_mask.float().mean() * 100
                self.log("train_confident_pixels_percent", confident_percent)
                self.log("current_confidence_threshold", current_threshold)
            
            # If we have confident pixels, compute consistency loss
            if confidence_mask.sum() > 0:
                # Student forward pass (with adaptation)
                student_logits = self(images_u)
                
                # Calculate consistency loss
                adapter_loss = self.consistency_loss(
                    student_logits,
                    teacher_logits,
                    confidence_mask
                )
                
                # Apply consistency weight
                weighted_adapter_loss = self.get_consistency_weight() * adapter_loss
                
                # Add to total loss
                total_loss += weighted_adapter_loss
                
                # Update EMA and log
                if self.consistency_loss_ema is None:
                    self.consistency_loss_ema = adapter_loss.detach().item()
                else:
                    self.consistency_loss_ema = self.ema_alpha * self.consistency_loss_ema + \
                                              (1 - self.ema_alpha) * adapter_loss.detach().item()
                
                # Log metrics
                self.log("train_adapter_loss", adapter_loss)
                self.log("train_adapter_loss_ema", self.consistency_loss_ema)
                self.log("train_consistency_weight", self.get_consistency_weight())
        
        # Log total loss
        self.log("train_total_loss", total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step evaluating the full model"""
        images, masks = batch
        
        # Full model evaluation
        logits = self(images)
        val_loss = self.supervised_loss(logits, masks)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Calculate IoU for each class
        iou_per_class = []
        for cls in range(self.num_classes):
            intersection = ((preds == cls) & (masks == cls)).float().sum()
            union = ((preds == cls) | (masks == cls)).float().sum()
            iou = intersection / (union + 1e-6)
            iou_per_class.append(iou)
        
        mean_iou = torch.stack(iou_per_class).mean()
        
        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_mean_iou", mean_iou, prog_bar=True)
        
        return val_loss
    
    def on_train_epoch_start(self):
        """Update current epoch counter"""
        self._current_epoch = self.trainer.current_epoch
    
    def configure_optimizers(self):
        """
        Configure optimizer with separate parameter groups for each component:
        1. Backbone (slowest learning rate)
        2. Adapter module (medium learning rate)
        3. Segmentation head (fastest learning rate)
        """
        # Separate parameters by component
        backbone_params = []
        adapter_params = []
        seghead_params = []
        
        # Group parameters
        for name, param in self.named_parameters():
            if 'feature_extractor' in name:
                backbone_params.append(param)
            elif 'adapter_module' in name:
                adapter_params.append(param)
            elif 'segmentation_head' in name:
                seghead_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.backbone_lr},
            {'params': adapter_params, 'lr': self.adapter_lr},
            {'params': seghead_params, 'lr': self.supervised_lr}
        ], weight_decay=self.weight_decay)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_total_loss",
                "interval": "epoch"
            }
        }
    
    def test_step(self, batch, batch_idx):
        """
        Run test step on a single batch
        """
        images, masks = batch
        
        # Forward pass
        logits = self(images)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, masks)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy
        accuracy = (preds == masks).float().mean()
        
        # Initialize metrics per class
        num_classes = logits.shape[1]
        metrics_per_class = []
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
        
        # Calculate metrics for each class and build confusion matrix
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix[i, j] = ((preds == j) & (masks == i)).sum()
        
        for c in range(num_classes):
            # True positives, false positives, false negatives
            true_pos = ((preds == c) & (masks == c)).sum().float()
            false_pos = ((preds == c) & (masks != c)).sum().float()
            false_neg = ((preds != c) & (masks == c)).sum().float()
            
            # Calculate metrics, avoiding division by zero
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0, device=self.device)
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0, device=self.device)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0, device=self.device)
            
            # Calculate IoU (Intersection over Union) for this class
            iou = true_pos / (true_pos + false_pos + false_neg) if (true_pos + false_pos + false_neg) > 0 else torch.tensor(0.0, device=self.device)
            
            metrics_per_class.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou
            })
        
        # Calculate overall metrics (micro average across all classes)
        total_true_pos = sum(confusion_matrix[i, i] for i in range(num_classes)).float()
        total_elements = confusion_matrix.sum().float()
        total_precision = total_true_pos / total_elements if total_elements > 0 else torch.tensor(0.0, device=self.device)
        total_f1 = 2 * total_precision * total_precision / (total_precision + total_precision) if total_precision > 0 else torch.tensor(0.0, device=self.device)
        
        # Return metrics to be accumulated
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'metrics_per_class': metrics_per_class,
            'total_f1': total_f1
        }
    
    def test_epoch_end(self, outputs):
        """
        Aggregate metrics from all test batches
        """
        num_classes = len(outputs[0]['metrics_per_class'])
        class_names = ["Background", "Vines"]  # Adjust based on your classes
        
        # Aggregate test loss and accuracy
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        
        # Aggregate confusion matrix
        total_confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
        for output in outputs:
            total_confusion_matrix += output['confusion_matrix']
        
        # Aggregate metrics per class
        avg_metrics_per_class = []
        for c in range(num_classes):
            avg_precision = torch.stack([x['metrics_per_class'][c]['precision'] for x in outputs]).mean()
            avg_recall = torch.stack([x['metrics_per_class'][c]['recall'] for x in outputs]).mean()
            avg_f1 = torch.stack([x['metrics_per_class'][c]['f1'] for x in outputs]).mean()
            avg_iou = torch.stack([x['metrics_per_class'][c]['iou'] for x in outputs]).mean()
            
            avg_metrics_per_class.append({
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'iou': avg_iou
            })
        
        # Calculate mean IoU across all classes
        mean_iou = torch.stack([metrics['iou'] for metrics in avg_metrics_per_class]).mean()
        
        # Calculate total F1 score (micro-average)
        total_f1 = torch.stack([x['total_f1'] for x in outputs]).mean()
        
        # Log metrics
        self.log('test_loss', avg_loss)
        self.log('test_accuracy', avg_accuracy)
        self.log('test_mean_iou', mean_iou)
        self.log('test_total_f1', total_f1)
        
        for c in range(num_classes):
            class_name = class_names[c] if c < len(class_names) else f"class_{c}"
            self.log(f'test_{class_name}_precision', avg_metrics_per_class[c]['precision'])
            self.log(f'test_{class_name}_recall', avg_metrics_per_class[c]['recall'])
            self.log(f'test_{class_name}_f1', avg_metrics_per_class[c]['f1'])
            self.log(f'test_{class_name}_iou', avg_metrics_per_class[c]['iou'])
        
        # Print results
        print("\n===== Test Set Evaluation =====")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Total F1 Score (micro-avg): {total_f1:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
        print("\nPer-class metrics:")
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"{class_name}:")
            print(f"  Precision: {avg_metrics_per_class[i]['precision']:.4f}")
            print(f"  Recall: {avg_metrics_per_class[i]['recall']:.4f}")
            print(f"  F1 Score: {avg_metrics_per_class[i]['f1']:.4f}")
            print(f"  IoU: {avg_metrics_per_class[i]['iou']:.4f}")
        
        print("\nConfusion Matrix:")
        print(total_confusion_matrix)
        
        # Return the metrics dictionary
        return {
            "test_loss": avg_loss,
            "test_accuracy": avg_accuracy,
            "test_total_f1": total_f1,
            "test_mean_iou": mean_iou,
            "test_confusion_matrix": total_confusion_matrix,
            "test_metrics_per_class": avg_metrics_per_class
        }
    
    def test(self, test_loader=None):
        """
        Comprehensive evaluation of segmentation model on test set
        This method can be called manually and provides a more organized output
        compared to trainer.test()
        
        Args:
            test_loader: Optional DataLoader for test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        if test_loader is None:
            raise ValueError("test_loader must be provided")
        
        self.eval()
        torch.set_grad_enabled(False)
        
        # Run test batches through test_step
        outputs = []
        for batch_idx, batch in enumerate(test_loader):
            batch = [x.to(self.device) for x in batch]
            output = self.test_step(batch, batch_idx)
            outputs.append(output)
        
        # Process results
        results = self.test_epoch_end(outputs)
        torch.set_grad_enabled(True)
        
        return results