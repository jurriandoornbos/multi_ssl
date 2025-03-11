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
from copy import deepcopy
from multissl.models.seghead import SegmentationModel


class MeanTeacherSegmentationModel(pl.LightningModule):
    """
    Mean Teacher model for semi-supervised segmentation.
    Implements the student-teacher paradigm where the teacher model is an exponential
    moving average (EMA) of the student model.
    """
    def __init__(
        self, 
        backbone_type="resnet50",
        pretrained_path=None,
        in_channels=4,
        num_classes=2, 
        img_size=224,
        lr=1e-4, 
        weight_decay=1e-5,
        consistency_weight=1.0,
        consistency_rampup=10,  # Epochs for ramping up consistency weight
        ema_decay=0.99,         # EMA decay rate for teacher model update
        class_weights=[1.0, 1.0],
        confidence_threshold=0.8,  # Threshold for confident pseudo-labels
        strong_threshold=0.2      # Threshold for strong augmentation intensity
    ):
        super().__init__()
        
        # Create student model
        self.student = SegmentationModel(
            backbone_type=backbone_type,
            pretrained_path=pretrained_path,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights
        )
        
        # Create teacher model (as a copy of student)
        self.teacher = deepcopy(self.student)
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self.confidence_threshold = confidence_threshold
        self.strong_threshold = strong_threshold
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights)
        
        # Current epoch tracking
        self._current_epoch = 0
        
        # For loss balancing - will be dynamically updated
        self.supervised_loss_scale = 1.0
        self.supervised_loss_ema = None  # Will track supervised loss with EMA
        self.consistency_loss_ema = None  # Will track consistency loss with EMA
        self.ema_alpha = 0.9  # EMA smoothing factor
        
        # Save model parameters
        self.save_hyperparameters(ignore=['student', 'teacher'])
    
    def forward(self, x):
        """Forward pass through student model"""
        return self.student(x)
    
    def teacher_forward(self, x):
        """Forward pass through teacher model (no gradient)"""
        with torch.no_grad():
            return self.teacher(x)
        
    def get_consistency_weight(self):
        """Ramps up the consistency weight during training"""
        # Linear ramp-up
        if self.consistency_rampup > 0:
            return self.consistency_weight * min(1.0, self._current_epoch / self.consistency_rampup)
        return self.consistency_weight
    
    def update_teacher(self):
        """Update teacher model using EMA of student model parameters"""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data
    
    def training_step(self, batch, batch_idx):
        """Training step supporting both labeled and unlabeled data with Mean Teacher approach"""
        labeled_batch, unlabeled_batch = batch
        
        # Initialize total_loss as a tensor to ensure it's always a tensor
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Track which losses are calculated for this batch
        has_supervised_loss = False
        has_consistency_loss = False
        
        # Process labeled data if we have non-empty tensors
        images_l, masks_l = labeled_batch
        if images_l.size(0) > 0:
            # Forward pass through student model for labeled data
            logits_l = self.student(images_l)
            supervised_loss = self.student.combined_loss(logits_l, masks_l)
            
            # Make sure supervised_loss is a tensor with gradient
            if isinstance(supervised_loss, torch.Tensor) and supervised_loss.requires_grad:
                # Update supervised loss EMA for loss balancing
                if self.supervised_loss_ema is None:
                    self.supervised_loss_ema = supervised_loss.item()
                else:
                    self.supervised_loss_ema = self.ema_alpha * self.supervised_loss_ema + (1 - self.ema_alpha) * supervised_loss.item()
                
                total_loss = supervised_loss  # Replace instead of adding to make sure it's a proper tensor
                self.log("train_supervised_loss", supervised_loss)
                has_supervised_loss = True
            else:
                # If supervised_loss isn't a proper tensor, create a new one
                self.log("warning", "supervised_loss is not a proper tensor")
                supervised_loss = torch.tensor(float(supervised_loss), device=self.device, requires_grad=True)
                total_loss = supervised_loss
                has_supervised_loss = True
        
        # Process unlabeled data if we have non-empty tensors
        if unlabeled_batch.size(0) > 0:
            images_u = unlabeled_batch
            
            # Generate pseudo labels with teacher model
            with torch.no_grad():
                teacher_logits = self.teacher(images_u)
                teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=1)
                max_probs, pseudo_masks = torch.max(teacher_probs, dim=1)
                
                # Create confidence mask for filtering uncertain predictions
                confidence_mask = (max_probs > self.confidence_threshold)
            
            # Only proceed if we have confident pixels
            if confidence_mask.sum() > 0:
                # Student model prediction on unlabeled data
                student_logits = self.student(images_u)
                
                # Apply consistency loss only to confident pixels
                consistency_loss = torch.nn.functional.cross_entropy(
                    student_logits, 
                    pseudo_masks, 
                    reduction='none'
                )
                
                # Apply confidence mask and calculate mean
                consistency_loss = (consistency_loss * confidence_mask.float()).sum() / confidence_mask.sum()
                
                # Ensure consistency_loss is a tensor with gradient
                if isinstance(consistency_loss, torch.Tensor) and consistency_loss.requires_grad:
                    # Update consistency loss EMA for loss balancing
                    if self.consistency_loss_ema is None:
                        self.consistency_loss_ema = consistency_loss.item()
                    else:
                        self.consistency_loss_ema = self.ema_alpha * self.consistency_loss_ema + (1 - self.ema_alpha) * consistency_loss.item()
                    
                    # Apply consistency weight with ramp-up
                    weighted_consistency_loss = self.get_consistency_weight() * consistency_loss
                    
                    # Scale consistency loss to be comparable to supervised loss
                    if self.supervised_loss_ema is not None and self.consistency_loss_ema is not None:
                        consistency_scale = self.supervised_loss_ema / (self.consistency_loss_ema + 1e-8)
                        scale_tensor = torch.clamp(torch.tensor(consistency_scale, device=self.device), 0.1, 10.0)
                        weighted_consistency_loss = weighted_consistency_loss * scale_tensor
                    
                    # Add to total loss if supervised loss exists, otherwise replace
                    if has_supervised_loss:
                        total_loss = total_loss + weighted_consistency_loss
                    else:
                        total_loss = weighted_consistency_loss
                    
                    self.log("train_consistency_loss", consistency_loss)
                    self.log("train_consistency_weight", self.get_consistency_weight())
                    has_consistency_loss = True
                    
                    # Log percentage of confident pixels
                    confident_pixel_percent = confidence_mask.float().mean() * 100
                    self.log("train_confident_pixels_percent", confident_pixel_percent)
        
        # Check if we have no losses (both batches empty or no confident pixels)
        if not has_supervised_loss and not has_consistency_loss:
            # Create a dummy loss to avoid errors
            # This will have zero gradient but prevents the training from breaking
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            return dummy_loss
        
        self.log("train_total_loss", total_loss)
        
        # Double-check that total_loss is a proper tensor
        if not isinstance(total_loss, torch.Tensor) or not total_loss.requires_grad:
            self.log("warning", "total_loss is not a proper tensor with gradient")
            # Convert to proper tensor if needed
            total_loss = torch.tensor(float(total_loss), device=self.device, requires_grad=True)
        
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher model after each batch"""
        self.update_teacher()
    
    def on_train_epoch_start(self):
        """Update current epoch counter"""
        self._current_epoch = self.trainer.current_epoch
    
    def validation_step(self, batch, batch_idx):
        """Validation step with metrics calculation"""
        # For validation, we assume a standard segmentation batch (images, masks)
        # without the semi-supervised structure
        images, masks = batch
        logits = self.student(images)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, masks)
        
        # Calculate metrics
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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_mean_iou", mean_iou, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler"""
        # Use separate parameter groups with different learning rates
        # for encoder (backbone) and decoder
        backbone_params = []
        decoder_params = []
        
        # Separate parameters
        for name, param in self.student.named_parameters():
            if 'feature_extractor' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.lr * 0.1},  # Lower LR for backbone
            {'params': decoder_params, 'lr': self.lr}          # Full LR for decoder
        ], weight_decay=self.weight_decay)
        
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
    
    def unfreeze_backbone(self, fine_tune_layers=None):
        """
        Unfreeze backbone layers for fine-tuning
        Args:
            fine_tune_layers: List of layer names to unfreeze, or None to unfreeze all
        """
        self.student.unfreeze_backbone(fine_tune_layers)

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