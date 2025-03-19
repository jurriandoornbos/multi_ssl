import torch
import torch.nn.functional as F
import numpy as np

def calculate_standard_error(model, test_loader, device='cuda'):
    """
    Calculate standard error of performance metrics across test batches.
    
    Args:
        model: The model to evaluate (can be teacher model for MeanTeacherSegmentation)
        test_loader: DataLoader for test data
        device: Device to run the evaluation on
        
    Returns:
        Dictionary containing mean and standard error for each metric
    """
    # Lists to store metrics from each batch
    batch_metrics = {
        'loss': [],
        'accuracy': [],
        'iou_per_class': [],
        'f1_per_class': [],
        'precision_per_class': [],
        'recall_per_class': []
    }
    
    model.eval()
    num_classes = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, masks = [x.to(device) for x in batch]
            
            # Get the number of classes from the first batch
            if num_classes is None:
                num_classes = model.student.num_classes if hasattr(model, 'student') else model.num_classes
                batch_metrics['iou_per_class'] = [[] for _ in range(num_classes)]
                batch_metrics['f1_per_class'] = [[] for _ in range(num_classes)]
                batch_metrics['precision_per_class'] = [[] for _ in range(num_classes)]
                batch_metrics['recall_per_class'] = [[] for _ in range(num_classes)]
            
            # Forward pass - use teacher for MeanTeacherSegmentation
            if hasattr(model, 'teacher'):
                logits = model.teacher(images)
            else:
                logits = model(images)
            
            # Calculate loss
            loss = F.cross_entropy(logits, masks)
            batch_metrics['loss'].append(loss.item())
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Calculate accuracy
            accuracy = (preds == masks).float().mean().item()
            batch_metrics['accuracy'].append(accuracy)
            
            # Calculate metrics for each class
            for c in range(num_classes):
                # True positives, false positives, false negatives
                true_pos = ((preds == c) & (masks == c)).sum().float()
                false_pos = ((preds == c) & (masks != c)).sum().float()
                false_neg = ((preds != c) & (masks == c)).sum().float()
                
                # Calculate metrics, avoiding division by zero
                if (true_pos + false_pos) > 0:
                    precision = (true_pos / (true_pos + false_pos)).item()
                else:
                    precision = 0.0
                    
                if (true_pos + false_neg) > 0:
                    recall = (true_pos / (true_pos + false_neg)).item()
                else:
                    recall = 0.0
                    
                if (precision + recall) > 0:
                    f1 = (2 * precision * recall / (precision + recall))
                else:
                    f1 = 0.0
                
                if (true_pos + false_pos + false_neg) > 0:
                    iou = (true_pos / (true_pos + false_pos + false_neg)).item()
                else:
                    iou = 0.0
                
                batch_metrics['iou_per_class'][c].append(iou)
                batch_metrics['f1_per_class'][c].append(f1)
                batch_metrics['precision_per_class'][c].append(precision)
                batch_metrics['recall_per_class'][c].append(recall)
    
    # Calculate mean and standard error for each metric
    results = {}
    
    # Process overall metrics
    for metric in ['loss', 'accuracy']:
        values = np.array(batch_metrics[metric])
        results[f'mean_{metric}'] = np.mean(values)
        results[f'std_error_{metric}'] = np.std(values) / np.sqrt(len(values))
        
    # Process per-class metrics
    for metric in ['iou_per_class', 'f1_per_class', 'precision_per_class', 'recall_per_class']:
        for c in range(num_classes):
            values = np.array(batch_metrics[metric][c])
            metric_name = metric.split('_per_class')[0]
            results[f'mean_{metric_name}_class{c}'] = np.mean(values)
            results[f'std_error_{metric_name}_class{c}'] = np.std(values) / np.sqrt(len(values))
    
    # Calculate mean IoU across all classes
    mean_iou_values = [np.mean(batch_metrics['iou_per_class'][c]) for c in range(num_classes)]
    results['mean_iou'] = np.mean(mean_iou_values)
    results['std_error_iou'] = np.std(mean_iou_values) / np.sqrt(len(mean_iou_values))
    
    # Calculate mean F1 across all classes
    mean_f1_values = [np.mean(batch_metrics['f1_per_class'][c]) for c in range(num_classes)]
    results['mean_f1'] = np.mean(mean_f1_values)
    results['std_error_f1'] = np.std(mean_f1_values) / np.sqrt(len(mean_f1_values))
    
    # Print results
    print("\n===== Performance Metrics with Standard Error =====")
    print(f"Loss: {results['mean_loss']:.4f} ± {results['std_error_loss']:.4f}")
    print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_error_accuracy']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f} ± {results['std_error_iou']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f} ± {results['std_error_f1']:.4f}")
    
    print("\nPer-class metrics:")
    class_names = ["Background", "Vines"]  # Adjust based on your classes
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{class_name}:")
        print(f"  Precision: {results[f'mean_precision_class{i}']:.4f} ± {results[f'std_error_precision_class{i}']:.4f}")
        print(f"  Recall: {results[f'mean_recall_class{i}']:.4f} ± {results[f'std_error_recall_class{i}']:.4f}")
        print(f"  F1 Score: {results[f'mean_f1_class{i}']:.4f} ± {results[f'std_error_f1_class{i}']:.4f}")
        print(f"  IoU: {results[f'mean_iou_class{i}']:.4f} ± {results[f'std_error_iou_class{i}']:.4f}")
    
    return results