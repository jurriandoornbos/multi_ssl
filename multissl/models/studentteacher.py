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
import numpy as np
from copy import deepcopy

from multissl.models.seghead import SegmentationModel


class MeanTeacherSegmentation(pl.LightningModule):
    """
    Mean Teacher model for semi-supervised semantic segmentation.
    
    This model maintains two networks:
    1. Student - updated directly through backpropagation
    2. Teacher - updated through Exponential Moving Average (EMA) of student weights
    
    The model uses consistency regularization between student and teacher predictions
    on unlabeled data to improve performance with limited labeled data.
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
        ema_decay=0.999,          # Decay rate for teacher EMA update
        consistency_weight=1.0,   # Weight for consistency loss
        consistency_rampup=40,    # Epochs to ramp up consistency weight
        confidence_threshold=0.8, # Confidence threshold for pseudo-labels
        class_weights=[1.0, 1.0],
        use_cutmix=True,          # Whether to use CutMix augmentation
        cutmix_prob=0.5,          # Probability of applying CutMix
        update_teacher_every=1    # Update teacher every N steps
    ):
        super().__init__()
        
        # Initialize student model
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
        
        # Initialize teacher model as a copy of student
        self.teacher = deepcopy(self.student)
        self._disable_teacher_grad()  # Teacher is not updated via backprop
        
        # Mean Teacher parameters
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self.confidence_threshold = confidence_threshold
        self.class_weights = torch.tensor(class_weights)
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.update_teacher_every = update_teacher_every
        
        # Optimization parameters
        self.lr = lr
        self.weight_decay = weight_decay
        
        # For tracking training progress
        self.step_count = 0
        self.current_consistency_weight = 0.0
                # For tracking training
        self._current_epoch = 0
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['student', 'teacher'])
    
    def _disable_teacher_grad(self):
        """Disable gradient computation for teacher model"""
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def update_teacher(self):
        """Update teacher weights using EMA of student weights"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=(1 - self.ema_decay)
                )
    
    def forward(self, x):
        """Forward pass using student model"""
        return self.student(x)
    
    def teacher_forward(self, x):
        """Forward pass using teacher model"""
        return self.teacher(x)
    
    def _get_current_consistency_weight(self):
        """Get consistency weight with linear ramp-up"""
        # Ramp up consistency weight linearly
        if self.consistency_rampup > 0:
            consistency_weight = self.consistency_weight * min(
                1.0, float(self._current_epoch) / self.consistency_rampup
            )
            return consistency_weight
        else:
            return self.consistency_weight
    
    def cutmix(self, image_batch, alpha=1.0):
        """
        Apply CutMix augmentation to batch of images
        
        Args:
            image_batch: Tensor of shape [B, C, H, W]
            alpha: Alpha parameter for beta distribution
            
        Returns:
            mixed_batch: Augmented batch of images
            lam: Lambda value for mixing (used for supervised loss)
            mix_indexes: Indexes used for mixing
        """
        batch_size, _, height, width = image_batch.size()
        
        # Generate random parameters for CutMix
        mix_indexes = torch.randperm(batch_size).to(image_batch.device)
        lam = np.random.beta(alpha, alpha)
        
        # Make sure lambda is not too extreme
        lam = max(lam, 1-lam)
        
        # Get random box coordinates
        cut_h = int(height * np.sqrt(1.0 - lam))
        cut_w = int(width * np.sqrt(1.0 - lam))
        
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        # Get corners of box
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        
        # Create mixed batch
        mixed_batch = image_batch.clone()
        mixed_batch[:, :, y1:y2, x1:x2] = image_batch[mix_indexes, :, y1:y2, x1:x2]
        
        return mixed_batch, lam, mix_indexes
    
    def consistency_loss(self, student_logits, teacher_logits, confidence_mask=None):
        """
        Calculate consistency loss between student and teacher predictions
        
        Args:
            student_logits: Predictions from student model
            teacher_logits: Predictions from teacher model
            confidence_mask: Optional binary mask of confident predictions
            
        Returns:
            Consistency loss
        """
        # Get teacher probabilities and detach
        teacher_probs = F.softmax(teacher_logits, dim=1).detach()
        
        # Get confidence scores from teacher
        confidence, _ = torch.max(teacher_probs, dim=1)
        
        # Create confidence mask if not provided
        if confidence_mask is None:
            confidence_mask = (confidence > self.confidence_threshold).float()
        
        # KL divergence loss (student learning from teacher distribution)
        student_log_probs = F.log_softmax(student_logits, dim=1)
        pixelwise_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction='none'
        ).sum(dim=1)
        
        # Apply confidence masking
        if confidence_mask.sum() > 0:
            # Only consider confident pixels
            masked_loss = (pixelwise_loss * confidence_mask).sum() / confidence_mask.sum()
        else:
            # Fall back to mean reduction if no pixels are confident
            masked_loss = pixelwise_loss.mean()
        
        return masked_loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step for Mean Teacher model
        
        The batch contains:
        - labeled_batch: (images, masks) for supervised learning
        - unlabeled_batch: images without masks for consistency regularization
        """
        labeled_batch, unlabeled_batch = batch
        labeled_images, labeled_masks = labeled_batch
        
        total_loss = 0.0
        
        # ---------- SUPERVISED TRAINING (LABELED DATA) ----------
        if labeled_images.size(0) > 0:
            # Forward through student model
            student_logits_l = self.student(labeled_images)
            
            # Calculate supervised loss (CE + Dice)
            supervised_loss = self.student.combined_loss(
                student_logits_l, labeled_masks
            )
            
            # Add supervised loss to total
            total_loss += supervised_loss
            
            # Log supervised loss
            self.log("train_supervised_loss", supervised_loss, prog_bar=True)
            
            # Also forward through teacher model for comparison (not used for loss)
            with torch.no_grad():
                teacher_logits_l = self.teacher(labeled_images)
                
                # Calculate teacher's performance (for monitoring)
                teacher_preds = torch.argmax(teacher_logits_l, dim=1)
                teacher_acc = (teacher_preds == labeled_masks).float().mean()
                
                # Log teacher accuracy
                self.log("train_teacher_acc", teacher_acc)
        
        # ---------- SEMI-SUPERVISED TRAINING (UNLABELED DATA) ----------
        if unlabeled_batch.size(0) > 0:
            unlabeled_images = unlabeled_batch
            
            # Apply CutMix augmentation with probability
            if self.use_cutmix and torch.rand(1).item() < self.cutmix_prob:
                # Mix unlabeled images
                mixed_unlabeled, _, _ = self.cutmix(unlabeled_images)
            else:
                mixed_unlabeled = unlabeled_images
            
            # Forward original unlabeled images through teacher (for pseudo-labels)
            with torch.no_grad():
                teacher_logits_u = self.teacher(unlabeled_images)
                
                # Get confidence scores
                teacher_probs = F.softmax(teacher_logits_u, dim=1)
                confidence, _ = torch.max(teacher_probs, dim=1)
                
                # Create confidence mask
                confidence_mask = (confidence > self.confidence_threshold)
                
                # Log percentage of confident pixels
                confident_percent = confidence_mask.float().mean() * 100
                self.log("train_confident_pixels_percent", confident_percent)
            
            # Forward mixed unlabeled images through student
            student_logits_u = self.student(mixed_unlabeled)
            
            # Calculate consistency loss
            consistency_loss = self.consistency_loss(
                student_logits_u, teacher_logits_u, confidence_mask
            )
            
            # Apply consistency weight with ramp-up
            current_weight = self._get_current_consistency_weight()
            weighted_consistency_loss = current_weight * consistency_loss
            
            # Add consistency loss to total
            total_loss += weighted_consistency_loss
            
            # Log consistency loss
            self.log("train_consistency_loss", consistency_loss)
            self.log("train_weighted_consistency_loss", weighted_consistency_loss)
            self.log("train_consistency_weight", current_weight, prog_bar=True)
        
        # Log total loss
        self.log("train_total_loss", total_loss, prog_bar=True)
        
        # Increment step counter
        self.step_count += 1
        
        # Update teacher model weights using EMA (every N steps)
        if self.step_count % self.update_teacher_every == 0:
            self.update_teacher()
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step using the teacher model
        
        Teacher model is used for validation as it should perform better
        due to the ensembling effect of EMA updates.
        """
        images, masks = batch
        
        # Forward through teacher model
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        # Calculate validation loss
        val_loss = F.cross_entropy(teacher_logits, masks)
        
        # Calculate metrics
        preds = torch.argmax(teacher_logits, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Calculate IoU for each class
        iou_per_class = []
        for cls in range(self.student.num_classes):
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
        """Configure optimizer and learning rate scheduler"""
        # Only optimize student model parameters
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
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
        logits = self.teacher(images)
        
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