import os
import torch
from datetime import datetime

# Global variables to track the log file
_log_file = None
_initialized = False

def initialize_roi_logger(log_dir="roi_logs", filename=None):
    """Initialize the logger with a file path"""
    global _log_file, _initialized
    
    os.makedirs(log_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"roi_log_{timestamp}.txt"
        
    _log_file = os.path.join(log_dir, filename)
    
    # Create or clear the log file
    with open(_log_file, 'w') as f:
        f.write(f"ROI Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    _initialized = True
    return _log_file  # Return the path for reference, not needed for usage

def write_to_log(message):
    """Write a message to the log file"""
    global _log_file, _initialized
    
    if not _initialized or _log_file is None:
        initialize_roi_logger()
        
    with open(_log_file, 'a') as f:
        f.write(message + "\n")

def log_proposal_stats(proposals, matched_idxs, labels, stage="unknown"):
    """Log statistics about proposals and their matching with GT boxes to file"""
    lines = [f"\n===== PROPOSAL STATS ({stage}) ====="]
    
    # Overall stats
    total_proposals = sum(len(p) for p in proposals)
    lines.append(f"Total proposals: {total_proposals}")
    
    # Per-image stats
    for i, (props, matches) in enumerate(zip(proposals, matched_idxs)):
        if isinstance(matches, list) and len(matches) == 0:
            lines.append(f"Image {i}: No matches (empty list)")
            continue
            
        if props.numel() == 0:
            lines.append(f"Image {i}: No proposals")
            continue
            
        # Count matches
        if torch.is_tensor(matches):
            pos_matches = (matches >= 0).sum().item()
            neg_matches = (matches == -1).sum().item()
            ignore_matches = (matches == -2).sum().item()
            
            lines.append(f"Image {i}: {len(props)} proposals")
            lines.append(f"  - Positive matches: {pos_matches}")
            lines.append(f"  - Negative matches: {neg_matches}")
            lines.append(f"  - Ignored matches: {ignore_matches}")
            
            # IoU distribution (if available)
            if hasattr(matches, 'get_match_quality_matrix'):
                iou_matrix = matches.get_match_quality_matrix()
                if iou_matrix is not None:
                    max_ious, _ = iou_matrix.max(dim=0)
                    lines.append(f"  - Max IoU (min/mean/max): {max_ious.min().item():.4f}/{max_ious.mean().item():.4f}/{max_ious.max().item():.4f}")
    
    # Check labels if available
    if labels is not None and torch.is_tensor(labels) and labels.numel() > 0:
        pos_labels = (labels > 0).sum().item()
        neg_labels = (labels == 0).sum().item()
        lines.append(f"\nLabels: {len(labels)} total")
        lines.append(f"  - Positive labels: {pos_labels}")
        lines.append(f"  - Negative labels: {neg_labels}")
        
        # Class distribution
        if pos_labels > 0:
            classes, counts = torch.unique(labels[labels > 0], return_counts=True)
            for cls, count in zip(classes.tolist(), counts.tolist()):
                lines.append(f"  - Class {cls}: {count} instances")
    
    # Write all lines to file
    write_to_log("\n".join(lines))

def log_target_stats(targets, stage="unknown"):
    """Log statistics about ground truth targets to file"""
    lines = [f"\n===== TARGET STATS ({stage}) ====="]
    
    if targets is None:
        lines.append("No targets provided")
        write_to_log("\n".join(lines))
        return
        
    # Overall stats
    total_boxes = sum(len(t["boxes"]) for t in targets)
    lines.append(f"Total GT boxes: {total_boxes}")
    
    # Per-image stats
    for i, target in enumerate(targets):
        boxes = target.get("boxes", None)
        labels = target.get("labels", None)
        masks = target.get("masks", None)
        
        if boxes is None or boxes.numel() == 0:
            lines.append(f"Image {i}: No GT boxes")
            continue
            
        lines.append(f"Image {i}: {len(boxes)} GT boxes")
        
        # Label distribution
        if labels is not None and labels.numel() > 0:
            classes, counts = torch.unique(labels, return_counts=True)
            for cls, count in zip(classes.tolist(), counts.tolist()):
                lines.append(f"  - Class {cls}: {count} instances")
        
        # Mask stats
        if masks is not None:
            lines.append(f"  - Masks: {len(masks)}")
            
            # Check mask validity
            if len(masks) > 0:
                non_empty = (masks.sum(dim=(1, 2)) > 0).sum().item()
                lines.append(f"  - Non-empty masks: {non_empty}/{len(masks)}")
    
    # Write all lines to file
    write_to_log("\n".join(lines))

def log_rpn_stats(objectness, matched_idxs, labels, regression_targets, stage="unknown"):
    """Log statistics about RPN outputs to file"""
    lines = [f"\n===== RPN STATS ({stage}) ====="]
    
    # Objectness scores
    if torch.is_tensor(objectness):
        obj_pos = (objectness > 0).float().mean().item()
        obj_mean = objectness.mean().item()
        obj_min = objectness.min().item()
        obj_max = objectness.max().item()
        
        lines.append(f"Objectness scores:")
        lines.append(f"  - Mean: {obj_mean:.4f}")
        lines.append(f"  - Min/Max: {obj_min:.4f}/{obj_max:.4f}")
        lines.append(f"  - Positive ratio: {obj_pos:.4f}")
        
        # Check for any NaN/Inf values
        nan_count = torch.isnan(objectness).sum().item()
        inf_count = torch.isinf(objectness).sum().item()
        if nan_count > 0 or inf_count > 0:
            lines.append(f"  - WARNING: Contains {nan_count} NaN and {inf_count} Inf values!")
    
    # Labels
    if isinstance(labels, list) and len(labels) > 0:
        try:
            all_labels = torch.cat(labels, dim=0)
            pos_count = (all_labels == 1).sum().item()
            neg_count = (all_labels == 0).sum().item()
            ignore_count = (all_labels == -1).sum().item()
            
            lines.append(f"Labels: {len(all_labels)} total")
            lines.append(f"  - Positive: {pos_count} ({pos_count/len(all_labels)*100:.2f}%)")
            lines.append(f"  - Negative: {neg_count} ({neg_count/len(all_labels)*100:.2f}%)")
            lines.append(f"  - Ignored: {ignore_count} ({ignore_count/len(all_labels)*100:.2f}%)")
        except Exception as e:
            lines.append(f"Error processing labels: {e}")
    
    # Write all lines to file
    write_to_log("\n".join(lines))

def log_subsample_stats(positive, negative, pos_idx, neg_idx, labels=None):
    """Log statistics about proposal sampling to file"""
    lines = [f"\n===== PROPOSAL SAMPLING STATS ====="]
    lines.append(f"Available positive samples: {positive.numel()}")
    lines.append(f"Available negative samples: {negative.numel()}")
    lines.append(f"Sampled positive samples: {len(pos_idx)}")
    lines.append(f"Sampled negative samples: {len(neg_idx)}")
    
    if labels is not None and torch.is_tensor(labels):
        try:
            sampled_indices = torch.cat([pos_idx, neg_idx]) if len(pos_idx) > 0 and len(neg_idx) > 0 else (
                pos_idx if len(pos_idx) > 0 else neg_idx
            )
            if len(sampled_indices) > 0:
                sampled_labels = labels[sampled_indices]
                pos_sampled = (sampled_labels > 0).sum().item()
                lines.append(f"Final positive samples: {pos_sampled}")
                lines.append(f"Final negative samples: {len(sampled_labels) - pos_sampled}")
        except Exception as e:
            lines.append(f"Error processing sampled labels: {e}")
    
    # Write all lines to file
    write_to_log("\n".join(lines))

def log_loss_stats(loss_dict):
    """Log loss values to file"""
    if not isinstance(loss_dict, dict):
        return
        
    lines = ["\n===== LOSS STATS ====="]
    for k, v in loss_dict.items():
        value = v.item() if hasattr(v, 'item') else v
        lines.append(f"{k}: {value:.6f}")
    
    # Write all lines to file
    write_to_log("\n".join(lines))