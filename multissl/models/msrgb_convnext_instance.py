import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, nms, batched_nms
import torch.optim as optim
from typing import Dict, Any, Tuple, List, Optional

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor

import math
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from .logging import *
from .msrgb_convnext import MSRGBConvNeXtFeatureExtractor

initialize_roi_logger()

class BatchedAnchorGenerator(nn.Module):
    """
    Anchor generator that works with batched tensor inputs instead of ImageList
    """
    def __init__(
        self,
        sizes: Tuple[Tuple[int, ...], ...] = ((128, 256, 512),),
        aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
    ):
        """
        Args:
            sizes: Tuple of tuples of anchor sizes for each feature map level
            aspect_ratios: Tuple of tuples of aspect ratios for each feature map level
        """
        super().__init__()
        
        if not isinstance(sizes[0], (list, tuple)):
            # Convert to nested tuple if single level provided
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            # Convert to nested tuple if single level provided  
            aspect_ratios = tuple((a,) for a in aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}
        
    def num_anchors_per_location(self) -> List[int]:
        """Number of anchors per spatial location for each feature map level"""
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
    
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """
        Generate anchor boxes for a single feature map level
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        # Generate base anchors centered at (0, 0)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()
    
    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        """Generate and cache cell anchors for all feature map levels"""
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # Assuming all cell_anchors have the same dtype and device
            if cell_anchors[0].device == device:
                return
        
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors
    
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Generate anchors for all spatial locations in the feature maps
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            
            # Generate all grid points
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
            # Add shifts to base anchors
            # shifts: (H*W, 4), base_anchors: (A, 4)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))
            
        return anchors
    
    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Generate anchors with caching for efficiency"""
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors
    
    def forward(self, image_batch: torch.Tensor, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Generate anchors for batched inputs
        
        Args:
            image_batch: Input images tensor (B, C, H, W)
            feature_maps: List of feature map tensors from FPN
            
        Returns:
            List of anchor tensors, flattened across all feature levels for each image
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_batch.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
                
        # Calculate strides for each feature map level
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        
        # Generate cell anchors if not already done
        self.set_cell_anchors(dtype, device)
        
        # Generate grid anchors for all feature map levels
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        
        # Concatenate anchors from all feature levels
        all_anchors = torch.cat(anchors_over_all_feature_maps, dim=0)
        
        # Replicate for each image in the batch
        batch_size = image_batch.shape[0]
        anchors = [all_anchors for _ in range(batch_size)]
        
        return anchors



class BatchedBoxCoder:
    """A more stable box coder for RoI heads"""
    def __init__(self, weights=(10.0, 10.0, 5.0, 5.0), 
                 bbox_xform_clip=math.log(1000. / 16),
                 image_size = (448,448)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
        self.image_size = image_size
        
    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to reference boxes
        with added numerical stability
        """
        # Convert to tensors if they're lists
        if isinstance(reference_boxes, list):
            reference_boxes = torch.cat(reference_boxes)
        if isinstance(proposals, list):
            proposals = torch.cat(proposals)
            
        # Handle empty tensors
        if reference_boxes.numel() == 0 or proposals.numel() == 0:
            return torch.zeros((0, 4), device=reference_boxes.device)
            
        # Ensure positive dimensions with a minimum size
        ex_widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
        ex_heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1.0)
        gt_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1.0)
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        
        # Apply weights and clip extreme values
        targets = targets * torch.tensor(self.weights, device=targets.device)
        targets = torch.clamp(targets, min=-4.0, max=4.0)
        
        # Check for NaN or Inf values and replace them
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            mask = torch.isnan(targets) | torch.isinf(targets)
            targets[mask] = 0.0
            print(f"WARNING: NaN or Inf in box regression targets: {mask.sum().item()} values replaced")
            
        return targets

    def decode(self, deltas, boxes, size=None):
        """
        Apply deltas to boxes with added numerical stability and error checking.
        This function is used during training but the box calculations are not 
        part of the gradient flow.
        
        Args:
            deltas: Bbox regression deltas (Nx4 tensor)
            boxes: Reference boxes (Nx4 tensor in [x1, y1, x2, y2] format)
            
        Returns:
            pred_boxes: Predicted boxes after applying deltas (Nx4 tensor)
        """
        # Since this function is used during training but boxes are not part
        # of gradient flow, we'll use torch.no_grad() for the box calculations
        # while preserving the original tensors
        
        # Convert to tensors if they're lists
        if isinstance(deltas, list):
            deltas = torch.cat(deltas)
        if isinstance(boxes, list):
            boxes = torch.cat(boxes)
        
        # Handle empty tensors
        if deltas.numel() == 0 or boxes.numel() == 0:
            return torch.zeros((0, 4), device=deltas.device)
        
        # Debug info - log statistics about inputs to help diagnose issues
        if torch.isnan(deltas).any() or torch.isinf(deltas).any():
            mask = torch.isnan(deltas) | torch.isinf(deltas)
            print(f"WARNING: NaN or Inf in box regression deltas: {mask.sum().item()} values replaced")
            deltas[mask] = 0.0
            
        # Debug info for boxes too
        if torch.isnan(boxes).any() or torch.isinf(boxes).any():
            mask = torch.isnan(boxes) | torch.isinf(boxes)
            print(f"WARNING: NaN or Inf in reference boxes: {mask.sum().item()} values detected")
            # Don't replace these - it indicates a deeper issue that should be fixed
        
        # Clone boxes to avoid in-place modification
        boxes = boxes.clone()
        
        # Validate box coordinates (boxes should have x2>x1, y2>y1)
        invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
        if invalid_boxes.any():
            print(f"WARNING: {invalid_boxes.sum().item()} invalid boxes detected with x2<=x1 or y2<=y1")
        
        # Ensure positive dimensions with small epsilon to avoid division by zero
        eps = 1e-5
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=eps)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=eps)
        
        # Compute box centers
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        # Apply normalization weights - ensure these match training values!
        # Common values are [1.0, 1.0, 1.0, 1.0] or [10.0, 10.0, 5.0, 5.0]
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = deltas[:, 2] / ww
        dh = deltas[:, 3] / wh
        
        # Clip width and height deltas to prevent extreme exponential values
        # Standard value is log(1000/16) ≈ 4.1
        dw = torch.clamp(dw, min=-self.bbox_xform_clip, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, min=-self.bbox_xform_clip, max=self.bbox_xform_clip)
        
        # Apply deltas to get predicted box centers and dimensions
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        # Convert from center format to x1,y1,x2,y2 format
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = (pred_ctr_x - 0.5 * pred_w).detach()  # x1
        pred_boxes[:, 1] = (pred_ctr_y - 0.5 * pred_h).detach()  # y1
        pred_boxes[:, 2] = (pred_ctr_x + 0.5 * pred_w).detach()  # x2
        pred_boxes[:, 3] = (pred_ctr_y + 0.5 * pred_h).detach()  # y2
        
        # Check for any remaining issues in the predicted boxes
        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            mask = torch.isnan(pred_boxes) | torch.isinf(pred_boxes)
            problem_indices = torch.where(mask.any(dim=1))[0]
            print(f"WARNING: NaN or Inf in predicted boxes: {mask.sum().item()} values")
            if len(problem_indices) > 0 and len(problem_indices) < 10:
                print(f"Problem at indices: {problem_indices.tolist()}")
                print(f"Original boxes at these indices: {boxes[problem_indices]}")
                print(f"Deltas at these indices: {deltas[problem_indices]}")
        
        # Optional: Clip to valid image coordinates if you have image dimensions available

        if size:
            h, w = size
            pred_boxes[:, 0].clamp_(min=0, max=w-1)
            pred_boxes[:, 1].clamp_(min=0, max=h-1)
            pred_boxes[:, 2].clamp_(min=0, max=w-1)
            pred_boxes[:, 3].clamp_(min=0, max=h-1)
            
        # Ensure x2 > x1 and y2 > y1
        pred_boxes[:, 2] = torch.max(pred_boxes[:, 2], pred_boxes[:, 0] + eps).detach()
        pred_boxes[:, 3] = torch.max(pred_boxes[:, 3], pred_boxes[:, 1] + eps).detach()
        
        return pred_boxes

class BatchedRegionProposalNetwork(nn.Module):
    """
    Region Proposal Network that works with batched tensor inputs
    """
    def __init__(
        self,
        anchor_generator,
        head,
        box_coder,
        fg_iou_thresh: float = 0.3,
        bg_iou_thresh: float = 0.7,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_top_n: Dict[str, int] = None,
        post_nms_top_n: Dict[str, int] = None,
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
        device: str = "cuda",
        image_size = None,
        
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = box_coder
        
        # Training parameters
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.device = device
        # Inference parameters
        if pre_nms_top_n is None:
            pre_nms_top_n = {"training": 2000, "testing": 1000}
        if post_nms_top_n is None:
            post_nms_top_n = {"training": 2000, "testing": 1000}
            
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """Compute RPN losses with robust NaN handling"""
        # Safely concatenate the labels


        if all(label.numel() > 0 for label in labels):
            all_labels = torch.cat(labels, dim=0)
        else:
            # Create a dummy tensor if we have empty labels
            return {
                "loss_objectness": torch.tensor(0.0, device=objectness.device),
                "loss_rpn_box_reg": torch.tensor(0.0, device=pred_bbox_deltas.device)
            }
        
        # Check for NaN in inputs
        if torch.isnan(objectness).any(): 
            print("WARNING: Found NaN in objectness scores")
            # Replace NaN/Inf with zeros
            objectness = torch.nan_to_num(objectness, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        # Check for NaN in inputs
        elif  torch.isinf(objectness).any():
            print("WARNING: Found Inf in objectness scores")
            # Replace NaN/Inf with zeros
            objectness = torch.nan_to_num(objectness, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Subsample labels (pos/neg)
        try:
            sampled_pos_inds, sampled_neg_inds = self.subsample_labels(labels)
            sampled_inds = torch.where(torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0))[0]
        except Exception as e:
            print(f"Error in subsample_labels: {e}")
            return {
                "loss_objectness": torch.tensor(0.0, device=objectness.device),
                "loss_rpn_box_reg": torch.tensor(0.0, device=pred_bbox_deltas.device)
            }
        
        # Flatten objectness
        objectness = objectness.flatten()
        
        # Safe BCE loss
        if sampled_inds.numel() > 0:
            # Use a try-except to handle potential errors
            try:
                objectness_loss = F.binary_cross_entropy_with_logits(
                    objectness[sampled_inds], 
                    all_labels[sampled_inds].float(),
                    reduction='none'
                )
                
                # Remove any NaN values before reduction
                valid_mask = ~torch.isnan(objectness_loss) & ~torch.isinf(objectness_loss)
                if valid_mask.sum() > 0:
                    objectness_loss = objectness_loss[valid_mask].mean()
                else:
                    objectness_loss = torch.tensor(0.0, device=objectness.device)
                    
            except Exception as e:
                print(f"Error in objectness loss: {e}")
                objectness_loss = torch.tensor(0.0, device=objectness.device)
        else:
            objectness_loss = torch.tensor(0.0, device=objectness.device)
        
        # Safe regression loss
        try:
            if isinstance(regression_targets, list) and len(regression_targets) > 0:
                regression_targets = torch.cat(regression_targets, dim=0)
            
            if sampled_pos_inds.numel() > 0:
                # Add a small epsilon to prevent division by zero
                regression_targets_pos = regression_targets[sampled_pos_inds]
                pred_deltas_pos = pred_bbox_deltas[sampled_pos_inds]
                
                # Replace any NaN/Inf values
                regression_targets_pos = torch.nan_to_num(regression_targets_pos, 
                                                        nan=0.0, posinf=0.0, neginf=0.0)
                pred_deltas_pos = torch.nan_to_num(pred_deltas_pos, 
                                                nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values 
                regression_targets_pos = torch.clamp(regression_targets_pos, min=-16.0, max=16.0)
                pred_deltas_pos = torch.clamp(pred_deltas_pos, min=-16.0, max=16.0)
                
                # Use robust smooth_l1_loss
                diff = pred_deltas_pos - regression_targets_pos
                diff_abs = diff.abs()
                smooth_l1_mask = diff_abs < 1.0
                loss = torch.where(
                    smooth_l1_mask, 
                    0.5 * diff * diff, 
                    diff_abs - 0.5
                )
                
                # Check for NaN/Inf in loss
                valid_mask = ~torch.isnan(loss) & ~torch.isinf(loss)
                if valid_mask.sum() > 0:
                    box_loss = loss[valid_mask].sum() / max(1, sampled_inds.numel())
                else:
                    box_loss = torch.tensor(0.0, device=pred_bbox_deltas.device)
            else:
                box_loss = torch.tensor(0.0, device=pred_bbox_deltas.device)
        except Exception as e:
            print(f"Error in regression loss: {e}")
            box_loss = torch.tensor(0.0, device=pred_bbox_deltas.device)
        
        return {"loss_objectness": objectness_loss, "loss_rpn_box_reg": box_loss}
    
    def subsample_labels(self, labels):
        """Subsample positive and negative labels for training"""
        pos_inds = []
        neg_inds = []
        
        for label in labels:
            pos_idx = torch.where(label == 1)[0]
            neg_idx = torch.where(label == 0)[0]
            
            # Number of positive samples
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(pos_idx.numel(), num_pos)
            
            # Number of negative samples
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(neg_idx.numel(), num_neg)
            
            # Random sampling
            perm_pos = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:num_pos]
            perm_neg = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]
            
            pos_inds.append(pos_idx[perm_pos])
            neg_inds.append(neg_idx[perm_neg])
            
        return torch.cat(pos_inds, dim=0), torch.cat(neg_inds, dim=0)
    
    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        """Filter proposals using NMS with robust error handling"""
        num_images = proposals.shape[0]
        device = proposals.device
        
        # Safely handle NaN/Inf in proposals and objectness
        if torch.isnan(proposals).any() or torch.isinf(proposals).any():
            print("WARNING: NaN/Inf in proposals, replacing with zeros")
            proposals = torch.nan_to_num(proposals, nan=0.0, posinf=1000.0, neginf=0.0)
        
        if torch.isnan(objectness).any() or torch.isinf(objectness).any():
            print("WARNING: NaN/Inf in objectness, replacing with zeros")
            objectness = torch.nan_to_num(objectness, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get scores and reshape
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        
        # Generate levels for batched NMS
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) 
                for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, dim=0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        
        final_boxes = []
        final_scores = []
        
        for i in range(num_images):
            # Work with per-image data for safety
            boxes = proposals[i]
            scores = objectness[i]
            lvl = levels[i]
            img_shape = image_shapes[i]
            
            # Select top-k proposals
            top_n = min(self.pre_nms_top_n["training" if self.training else "testing"], 
                        scores.shape[0])
            
            # Handle potential empty tensors
            if scores.shape[0] == 0:
                final_boxes.append(torch.zeros((0, 4), device=device))
                final_scores.append(torch.zeros((0,), device=device))
                continue
            
            # Safe topk operation
            try:
                scores_for_topk = scores.clone()
                _, idx = scores_for_topk.topk(top_n, sorted=True)
                boxes_topk = boxes[idx]
                scores_topk = scores[idx]
                lvl_topk = lvl[idx]
                
                # Clip boxes safely (create new tensor)
                boxes_clipped = torch.zeros_like(boxes_topk)
                boxes_clipped[:, 0] = boxes_topk[:, 0].clamp(min=0, max=img_shape[1])
                boxes_clipped[:, 1] = boxes_topk[:, 1].clamp(min=0, max=img_shape[0])
                boxes_clipped[:, 2] = boxes_topk[:, 2].clamp(min=0, max=img_shape[1])
                boxes_clipped[:, 3] = boxes_topk[:, 3].clamp(min=0, max=img_shape[0])
                
                # Remove small boxes safely
                widths = boxes_clipped[:, 2] - boxes_clipped[:, 0]
                heights = boxes_clipped[:, 3] - boxes_clipped[:, 1]
                keep = (widths > 1e-3) & (heights > 1e-3)
                
                # Check if we have valid boxes left
                if keep.sum() == 0:
                    final_boxes.append(torch.zeros((0, 4), device=device))
                    final_scores.append(torch.zeros((0,), device=device))
                    continue
                
                boxes_filtered = boxes_clipped[keep]
                scores_filtered = scores_topk[keep]
                lvl_filtered = lvl_topk[keep]
                
                # Safe NMS
                try:
                    keep_nms = batched_nms(boxes_filtered, scores_filtered, lvl_filtered, self.nms_thresh)
                    
                    # Select post-NMS top-k
                    post_nms_top_n = self.post_nms_top_n["training" if self.training else "testing"]
                    keep_nms = keep_nms[:post_nms_top_n]
                    
                    final_boxes.append(boxes_filtered[keep_nms])
                    final_scores.append(scores_filtered[keep_nms])
                except Exception as e:
                    print(f"Error in NMS: {e}")
                    final_boxes.append(torch.zeros((0, 4), device=device))
                    final_scores.append(torch.zeros((0,), device=device))
                    
            except Exception as e:
                print(f"Error in proposal filtering: {e}")
                final_boxes.append(torch.zeros((0, 4), device=device))
                final_scores.append(torch.zeros((0,), device=device))
        
        return final_boxes, final_scores
    
    def remove_small_boxes(self, boxes, min_size):
        """Remove boxes with width or height less than min_size"""
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        return torch.where(keep)[0]
            
    def box_similarity(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes with improved numerical stability
        and thorough error handling
        
        Args:
            boxes1: First set of boxes, shape (N, 4) in [x1, y1, x2, y2] format
            boxes2: Second set of boxes, shape (M, 4) in [x1, y1, x2, y2] format
            
        Returns:
            IoU matrix of shape (N, M)
        """
        # Handle empty boxes
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
        
        # Ensure boxes are properly formed (x2 > x1, y2 > y1)
        # Apply a small epsilon to ensure positive areas
        eps = 1e-5
        
        # Safely compute areas with clamping to ensure positive values
        width1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=eps)
        height1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=eps)
        width2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=eps)
        height2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=eps)
        
        area1 = width1 * height1
        area2 = width2 * height2
        
        # Compute intersection coordinates
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]
        
        # Compute intersection area with safety checks
        wh = (rb - lt).clamp(min=0)  # width-height [N,M,2]
        intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        # Compute union area with safety for division
        union = area1[:, None] + area2 - intersection
        
        # Add epsilon to denominator to prevent division by zero
        union = union.clamp(min=eps)
        
        # Compute IoU
        iou = intersection / union
        
        # Safety check for NaN/Inf values (shouldn't happen with above safeguards)
        if torch.isnan(iou).any() or torch.isinf(iou).any():
            # Create a mask for valid values
            valid_mask = ~(torch.isnan(iou) | torch.isinf(iou))
            
            # Create a new tensor with zeros where values were invalid
            valid_iou = torch.zeros_like(iou)
            valid_iou[valid_mask] = iou[valid_mask]
            
            # Report the issue
            invalid_count = (~valid_mask).sum().item()
            print(f"WARNING: Fixed {invalid_count} NaN/Inf values in IoU calculation")
            
            return valid_iou
        
        return iou

    def assign_targets_to_anchors(self, anchors, targets):
        """Assign ground truth targets to anchors for training with empty tensor handling"""
        labels = []
        matched_gt_boxes = []
        
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            
            # Handle empty ground truth case
            if gt_boxes.shape[0] == 0:
                labels_per_image = torch.zeros(anchors_per_image.shape[0], 
                                            dtype=torch.long, 
                                            device=anchors_per_image.device)
                matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image)
            else:
                # Compute IoU matrix with empty tensor handling
                iou_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                
                # Assign based on max IoU
                matched_vals, matched_idxs = iou_matrix.max(dim=0)
                
                # Create labels
                labels_per_image = torch.zeros_like(matched_vals, dtype=torch.long)
                labels_per_image[matched_vals >= self.fg_iou_thresh] = 1
                labels_per_image[matched_vals < self.bg_iou_thresh] = 0
                
                matched_gt_boxes_per_image = gt_boxes[matched_idxs]
                
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            
        return labels, matched_gt_boxes
    
    def forward(self, images, features, targets=None):
        """
        Forward pass through RPN
        
        Args:
            images: Input images tensor (B, C, H, W)
            features: Dict of feature maps from FPN
            targets: Training targets (optional)
            
        Returns:
            proposals: List of proposal boxes per image
            losses: Dict of losses (during training)
        """
        # Convert features dict to list for compatibility
        feature_maps = list(features.values())
        b,c,h,w = images.shape

        # Generate anchors
        anchors = self.anchor_generator(images, feature_maps)
        
        # Get number of anchors per level for each image
        num_anchors_per_level = [x.shape[0] for x in self.anchor_generator.grid_anchors(
            [f.shape[-2:] for f in feature_maps],
            [[torch.tensor(images.shape[-2] // f.shape[-2]), torch.tensor(images.shape[-1] // f.shape[-1])] 
             for f in feature_maps]
        )]
        
        # Run RPN head
        objectness, pred_bbox_deltas = self.head(feature_maps)
        
        # Concatenate predictions across all levels
        objectness = torch.cat([o.permute(0, 2, 3, 1).reshape(images.shape[0], -1) for o in objectness], dim=1)
        pred_bbox_deltas = torch.cat([p.permute(0, 2, 3, 1).reshape(images.shape[0], -1, 4) for p in pred_bbox_deltas], dim=1)
        
        # Decode proposals
        # We need to repeat anchors for each image in the batch
        batch_size = images.shape[0]
        anchor_tensor = anchors[0]  # Get the anchor tensor (same for all images)
        
        # Expand anchors to match batch size
        num_anchors = anchor_tensor.shape[0]
        expanded_anchors = anchor_tensor.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 4)
        
        # Decode proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.view(-1, 4), expanded_anchors, size = (h,w))
        proposals = proposals.view(batch_size, num_anchors, 4)
        
        # Get image shapes
        image_shapes = [images.shape[-2:] for _ in range(images.shape[0])]
        
        # Filter proposals
        boxes, scores = self.filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)
        
        losses = {}
        if self.training and targets is not None:
            # Assign targets to anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            
            # Compute regression targets
            regression_targets = []
            for matched_boxes, anchor_per_image in zip(matched_gt_boxes, anchors):
                targets = self.box_coder.encode(matched_boxes, anchor_per_image)
                regression_targets.append(targets)
            
            # Compute losses
            losses = self.compute_loss(objectness, pred_bbox_deltas.view(-1, 4), labels, regression_targets)
        
        return boxes, losses
    

    
# Class for improved mask predictor to handle class imbalance
class ImprovedMaskPredictor(MaskRCNNPredictor):
   """
   Enhanced mask predictor with better handling of boundary regions
   """
   def __init__(self, in_channels, dim_reduced, num_classes):
       super().__init__(in_channels, dim_reduced, num_classes)
       
       # Replace standard layers with more robust ones
       self.conv5_mask = nn.Sequential(
           nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1),
           nn.GroupNorm(32, dim_reduced),
           nn.ReLU(inplace=True),
           nn.Conv2d(dim_reduced, num_classes, 1)
       )
          
   def forward(self, x):
       """
       Forward pass with improved numerical stability
       """
       # Apply convolutions with better feature extraction
       x = self.conv5_mask(x)
       
       # Clamp extreme values for stability
       return torch.clamp(x, min=-15.0, max=15.0)
   
class ImprovedMaskHead(MaskRCNNHeads):
    """
    Enhanced mask head with better stability and performance
    """
    def __init__(self, in_channels, layers, dilation=1):
        super().__init__(in_channels, layers, dilation)
        
        # Initialize with attention mechanism for better feature focus
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add layer normalization for better stability
        self.layer_norms = nn.ModuleList()
        
        # Directly recreate the convolution layers to make sure we have access to them
        # This avoids needing to access the parent class's private attributes
        self.conv_layers_list = nn.ModuleList()
        
        for layer_idx, layer_features in enumerate(layers):
            conv = nn.Conv2d(
                in_channels if layer_idx == 0 else layers[layer_idx - 1],
                layer_features,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation
            )
            self.conv_layers_list.append(conv)
            
            # Add layer norm for all but the last layer
            if layer_idx < len(layers) - 1:
                self.layer_norms.append(nn.GroupNorm(32, layer_features))
        
    def forward(self, x):
        """
        Forward pass with attention mechanism and better normalization
        """
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Pass through convolutions with layer norm
        for i, layer in enumerate(self.conv_layers_list):
            x = layer(x)
            
            # Apply layer norm and ReLU except for last layer
            if i < len(self.conv_layers_list) - 1:
                x = self.layer_norms[i](x)
                x = F.relu(x, inplace=True)
                
        return x    

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network implementation
    Creates top-down pathway with lateral connections
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            # 1x1 conv for lateral connections
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            # 3x3 conv for final feature maps
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        
                
    def forward(self, features):
        """
        Args:
            features: OrderedDict of feature maps from backbone
        Returns:
            OrderedDict of FPN feature maps
        """
        # Get feature maps as list (bottom-up order)
        feature_list = list(features.values())
        
        # Start from the deepest feature map
        inner_lateral = self.inner_blocks[-1](feature_list[-1])
        result = [self.layer_blocks[-1](inner_lateral)]
        
        # Top-down pathway with lateral connections
        for idx in range(len(feature_list) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](feature_list[idx])
            
            # Upsample and add
            inner_top_down = F.interpolate(
                result[0], 
                size=inner_lateral.shape[-2:], 
                mode='nearest'
            )
            inner_lateral = inner_lateral + inner_top_down
            
            # Apply 3x3 conv
            result.insert(0, self.layer_blocks[idx](inner_lateral))
        
        # Convert back to OrderedDict with proper naming
        fpn_features = {}
        feature_names = list(features.keys())
        for i, feat in enumerate(result):
            fpn_features[feature_names[i]] = feat
            
        return fpn_features

    
class MultiLevelRPNHead(nn.Module):
    """
    RPN Head that operates on multiple FPN levels
    """
    def __init__(self, in_channels, num_anchors_per_location):
        super().__init__()
        # Shared convolution across all levels
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        # Classification head (object/background)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors_per_location, 1)
        
        # Regression head (bbox deltas)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors_per_location * 4, 1)
                
    def forward(self, features):
        """
        Args:
            features: List of feature maps from FPN
        Returns:
            logits: List of classification scores for each level
            bbox_reg: List of bbox regression for each level
        """
        logits = []
        bbox_reg = []
        
        for feature in features:
            x = F.relu(self.conv(feature))
            logits.append(self.cls_logits(x))
            bbox_reg.append(self.bbox_pred(x))
            
        return logits, bbox_reg

class ImprovedRoIHeads(nn.Module):
    def __init__(
        self,
        box_feature_extractor,
        box_predictor,
        box_coder,
        box_roi_pool,
        mask_feature_extractor,
        mask_predictor,
        mask_roi_pool,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        use_matched_boxes=False,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        mask_roi_output_size = 28,
    ):
        """
        RoI head for Mask R-CNN handling batched inputs.
        
        Args:
            box_feature_extractor: backbone network to extract box features
            box_predictor: module that predicts bounding boxes and classes
            mask_feature_extractor: backbone to extract mask features
            mask_predictor: module that predicts masks
            fg_iou_thresh: IoU threshold for foreground
            bg_iou_thresh: IoU threshold for background
            batch_size_per_image: number of proposals per image
            positive_fraction: fraction of positive (foreground) proposals
            bbox_reg_weights: weights for bbox regression
            use_matched_boxes: use matched gt boxes as proposals
            score_thresh: score threshold for predictions
            nms_thresh: NMS threshold
            detections_per_img: max number of detections per image
        """
        super(ImprovedRoIHeads, self).__init__()
        
        self.box_feature_extractor = box_feature_extractor
        self.box_predictor = box_predictor
        self.mask_feature_extractor = mask_feature_extractor
        self.mask_predictor = mask_predictor
        
        # IoU thresholds for assigning foreground and background
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_matched_boxes = use_matched_boxes
        
        # NMS and detection thresholds
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.mask_roi_output_size = mask_roi_output_size
        if bbox_reg_weights is None:
            # Default regression weights (x, y, w, h)
            bbox_reg_weights = (1.0,1.0,1.0,1.0)
        self.box_coder = box_coder
        
        # RoI alignment for box and mask features
        self.box_roi_pool = box_roi_pool
        
        self.mask_roi_pool = mask_roi_pool

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Forward pass supporting batched inputs.
        
        Args:
            features: Dict[str, Tensor], output of the backbone
            proposals: List[Tensor[N, 4]], proposed regions (x1, y1, x2, y2)
            image_shapes: List[Tuple[H, W]], original image shapes
            targets: optional List[Dict], ground truth annotations
                
        Returns:
            detections: List[Dict], detected objects
            losses: Dict, losses during training
        """
        if self.training:
            # During training, we need ground truth targets
            if targets is None:
                raise ValueError("In training mode, targets should be passed")

          # Assign target boxes to proposed regions
            proposals, matched_idxs, labels, regression_targets = self.assign_targets_to_proposals(
                proposals, targets
            )
                
        # Get box features from proposals using RoIAlign
        box_features = self.extract_box_features(features, proposals)
        # Inside the forward method of ImprovedRoIHeads, where you predict class and box regression:

        # Check if we have any proposals
        if box_features.numel() == 0:
            # Handle completely empty batch
            result = [{"boxes": torch.empty((0, 4), device=features["0"].device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=features["0"].device),
                    "scores": torch.empty((0,), device=features["0"].device)}
                    for _ in range(len(proposals))]
            losses = {"loss_classifier": torch.tensor(0.0, device=features["0"].device),
                    "loss_box_reg": torch.tensor(0.0, device=features["0"].device)}
            if self.mask_predictor is not None:
                losses["loss_mask"] = torch.tensor(0.0, device=features["0"].device)
                for r in result:
                    r["masks"] = torch.empty((0, 1, self.mask_roi_output_size, self.mask_roi_output_size), device=features["0"].device)
            return result, losses
        
        # Predict box classification and regression
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        
        if self.training:
            # Calculate classification and regression losses
            loss_classifier, loss_box_reg = self.compute_box_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
        else:
            # During inference, apply classification and regression to get final boxes
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
                
            # Create result list for each image in batch
            num_images = len(boxes)
            result = []
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "scores": scores[i],
                        "labels": labels[i],
                    }
                )
        
        # Process masks if we have detections and mask predictor exists
        if self.mask_predictor is not None:
            if self.training:
                # During training, get mask targets using efficient matching
                sampled_pos_inds_subset = torch.where(labels > 0)[0]
                
                # Fix: Check if we have any positive samples
                if sampled_pos_inds_subset.numel() == 0:
                    # No positive samples - add zero mask loss
                    losses["loss_mask"] = box_regression.sum() * 0.0
                    return result, losses
                    
                # Get positive proposals for mask prediction (with improved filtering)
                pos_proposals = []
                start_idx = 0
                for p in proposals:
                    end_idx = start_idx + p.shape[0]
                    # Find which positive indices fall within this image's proposals
                    img_pos_indices = sampled_pos_inds_subset[(sampled_pos_inds_subset >= start_idx) & 
                                                            (sampled_pos_inds_subset < end_idx)]
                    # Convert to local indices within the image
                    img_pos_indices = img_pos_indices - start_idx
                    
                    # Filter to keep only top-K positive proposals per image (e.g., top 100)
                    if len(img_pos_indices) > 100:
                        # Get scores for these proposals (if available)
                        # Or use IoU with GT as a proxy for quality
                        _, sorted_indices = torch.sort(labels[img_pos_indices], descending=True)
                        img_pos_indices = img_pos_indices[sorted_indices[:100]]
                    
                    pos_proposals.append(p[img_pos_indices])
                    start_idx = end_idx
                
                # Get mask features and targets
                mask_features = self.extract_mask_features(features, pos_proposals)
                
                # Fix: Handle empty case
                if mask_features.numel() == 0:
                    losses["loss_mask"] = box_regression.sum() * 0.0
                    return result, losses
                    
                # Get mask targets with improved matching
                mask_targets = self.get_mask_targets(pos_proposals, targets, matched_idxs, labels)
                    
                # Predict masks and compute loss
                mask_logits = self.mask_predictor(mask_features)
                
                # Fix dimensional mismatch in compute_mask_loss
                if len(mask_logits.shape) == 4 and len(mask_targets.shape) == 3:
                    # If binary mask prediction (foreground/background)
                    if mask_logits.shape[1] == 2:
                        # Use the foreground channel (index 1)
                        mask_logits = mask_logits[:, 1]
                    else:
                        # For multi-class case, expand targets
                        mask_targets = mask_targets.unsqueeze(1)
                
                loss_mask = F.binary_cross_entropy_with_logits(
                    mask_logits,
                    mask_targets.to(dtype=torch.float32),
                    reduction="mean",
                )
                losses["loss_mask"] = loss_mask
            else:
                # During inference, limit mask prediction to top-K boxes by score
                MAX_MASKS_PER_IMAGE = 100  # Configurable parameter
                
                # Filter boxes for mask prediction (top-K by score)
                mask_boxes = []
                for r in result:
                    if len(r["boxes"]) > 0:
                        scores = r["scores"]
                        sorted_idxs = torch.argsort(scores, descending=True)
                        top_k_idxs = sorted_idxs[:min(MAX_MASKS_PER_IMAGE, len(sorted_idxs))]
                        mask_boxes.append(r["boxes"][top_k_idxs])
                        
                        # Update result to include only these boxes for masks
                        if "masks" in r:
                            r["mask_indices"] = top_k_idxs  # Save indices for later
                    else:
                        mask_boxes.append(torch.zeros((0, 4), device=features["0"].device))
                
                # Check if we have any boxes to predict masks for
                if sum(len(p) for p in mask_boxes) > 0:
                    mask_features = self.extract_mask_features(features, mask_boxes)
                    mask_logits = self.mask_predictor(mask_features)
                        
                    # Convert mask logits to binary masks
                    if mask_logits.shape[1] == 2:
                        # For binary case, use only foreground channel
                        mask_probs = mask_logits[:, 1].sigmoid()
                    else:
                        # For multi-class case
                        mask_probs = mask_logits.sigmoid()
                    
                    # Split mask predictions per image
                    mask_probs = mask_probs.split([len(p) for p in mask_boxes], dim=0)
                        
                    # Add masks to result (only for filtered boxes)
                    for i, (mask_prob, r) in enumerate(zip(mask_probs, result)):
                        if "mask_indices" in r:
                            # Create full-sized mask tensor with zeros
                            full_masks = torch.zeros((len(r["boxes"]), 1, self.mask_roi_output_size, 
                                                self.mask_roi_output_size), device=mask_prob.device)
                            
                            # Place computed masks in their correct positions
                            if len(mask_prob) > 0:
                                if len(mask_prob.shape) == 3:  # Add channel dim if needed
                                    mask_prob = mask_prob.unsqueeze(1)
                                full_masks[r["mask_indices"]] = mask_prob
                                
                            r["masks"] = full_masks
                            del r["mask_indices"]  # Clean up temporary indices
                        else:
                            # For images with no detections
                            r["masks"] = torch.zeros((0, 1, self.mask_roi_output_size, self.mask_roi_output_size), 
                                                device=features["0"].device)
                else:
                    # No boxes detected, add empty masks
                    for r in result:
                        r["masks"] = torch.zeros((0, 1, self.mask_roi_output_size, self.mask_roi_output_size), 
                                            device=features["0"].device)
        
        return result, losses
    
    def get_mask_targets(self, proposals, targets, matched_idxs, labels):
        """
        Get mask targets for each positive proposal with improved matching.
        """
        device = labels.device
        
        # Count the number of positive samples per image
        pos_count = sum(len(p) for p in proposals)
        
        # Handle empty case
        if pos_count == 0:
            return torch.zeros((0, self.mask_roi_output_size, self.mask_roi_output_size), device=device)
        
        # Process each image
        mask_targets = []
        proposal_count = 0
        proposal_idx_offset = 0
        
        # Process each image
        for img_idx, (img_proposals, img_targets) in enumerate(zip(proposals, targets)):
            # Skip images with no positive proposals
            if len(img_proposals) == 0:
                continue
                
            # Get ground truth masks
            gt_masks = img_targets.get("masks", None)
            if gt_masks is None or len(gt_masks) == 0:
                # No masks for this image, create empty masks
                img_mask_targets = torch.zeros((len(img_proposals), self.mask_roi_output_size, self.mask_roi_output_size), 
                                            device=device)
                mask_targets.append(img_mask_targets)
                proposal_count += len(img_proposals)
                proposal_idx_offset += len(img_proposals)
                continue
            
            # Get matched indices for this image
            start_idx = proposal_idx_offset
            end_idx = start_idx + len(img_proposals)
            img_matched_idxs_indices = torch.arange(start_idx, end_idx, device=device)
            img_matched_idxs = matched_idxs[img_idx]
            
            # Process each proposal
            img_mask_targets = []
            for i, prop in enumerate(img_proposals):
                # Get corresponding ground truth mask index
                gt_idx = img_matched_idxs[i]
                
                # Skip if no match - FIXED LINE
                if gt_idx < 0 or gt_idx >= len(gt_masks):
                    empty_mask = torch.zeros((self.mask_roi_output_size, self.mask_roi_output_size), device=device)
                    img_mask_targets.append(empty_mask)
                    continue
                    
                # Get matched GT mask and resize to fixed size
                gt_mask = gt_masks[gt_idx]
                resized_mask = F.interpolate(
                    gt_mask.float().unsqueeze(0).unsqueeze(0), 
                    size=(self.mask_roi_output_size, self.mask_roi_output_size), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0).squeeze(0)
                
                img_mask_targets.append(resized_mask)
            
            # Combine masks for this image
            if img_mask_targets:
                img_mask_targets = torch.stack(img_mask_targets)
                mask_targets.append(img_mask_targets)
                proposal_count += len(img_proposals)
            
            proposal_idx_offset += len(img_proposals)
        
        # Combine mask targets from all images
        if mask_targets:
            return torch.cat(mask_targets, dim=0)
        else:
            return torch.zeros((0, self.mask_roi_output_size, self.mask_roi_output_size), device=device)
    def extract_mask_features(self, features, proposals):
        """
        Extract mask features using RoIAlign.
        """
        # Convert dict of features to list
        features_list = [features[k] for k in sorted(features.keys())]
        
        # Apply RoI pooling to get fixed-size features
        mask_features = self.mask_roi_pool(features, proposals, [img.shape for img in features_list[:1]])
        
        # Process features through mask feature extractor
        mask_features = self.mask_feature_extractor(mask_features)
        
        return mask_features
        
    def subsample_proposals(self, labels):
        """
        Sample proposals for training with a fixed positive/negative ratio.
        Handles cases where there are no positive or negative samples safely.
        """
        # Safety check for empty labels
        if labels.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=labels.device)
            
        # Find positive and negative indices
        positive = torch.where(labels > 0)[0]
        negative = torch.where(labels == 0)[0]
        
        # Determine number of positive/negative samples
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(negative.numel(), num_neg)
        
        # Handle edge case: no positives
        if positive.numel() == 0:
            # Just return negative samples
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            return negative[perm2]
            
        # Handle edge case: no negatives
        if negative.numel() == 0:
            # Just return positive samples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            return positive[perm1]
        
        # Randomly sample positive and negative proposals
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        
        pos_idx = positive[perm1]
        neg_idx = negative[perm2]
        
        # Combine indices
        return torch.cat([pos_idx, neg_idx])
    
    def compute_box_loss(self, class_logits, box_regression, labels, regression_targets):
        """
        Compute classification and box regression losses.
        Properly handles empty batches and ensures tensor shape consistency.
        """
        # Classification loss (cross entropy)
        classification_loss = F.cross_entropy(class_logits, labels)
        
        # Get positive examples
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        
        # Handle empty positive samples
        if sampled_pos_inds_subset.numel() == 0:
            # Return zero box loss if no positive samples
            return classification_loss, box_regression.sum() * 0.0
        
        labels_pos = labels[sampled_pos_inds_subset]
        
        # Calculate box regression correctly
        N, num_regression_values = box_regression.shape
        
        # Check if regression output includes background class
        # FastRCNNPredictor typically outputs regression values only for non-background classes
        num_classes = class_logits.shape[1]  # This includes background
        num_reg_classes = num_regression_values // 4
        
        # If regression doesn't include background class, offset labels by 1
        box_regression_flat = box_regression.reshape(N, -1, 4)
        if num_reg_classes == num_classes - 1:  # Background not included in regression
            class_idx = labels_pos - 1  # Offset by 1 to skip background
        else:  # Background included in regression
            class_idx = labels_pos
        
        box_regression_pos = box_regression_flat[sampled_pos_inds_subset, class_idx]
        regression_targets_pos = regression_targets[sampled_pos_inds_subset]
        
        # Ensure shapes match
        if box_regression_pos.shape != regression_targets_pos.shape:
            print(f"Shape mismatch: {box_regression_pos.shape} vs {regression_targets_pos.shape}")
            
            # Ensure shapes match by reshaping if needed
            if box_regression_pos.shape[0] == regression_targets_pos.shape[0]:
                regression_targets_pos = regression_targets_pos.reshape(box_regression_pos.shape)
        
        # Box regression loss (smooth L1)
        box_loss = F.smooth_l1_loss(
            box_regression_pos,
            regression_targets_pos,
            reduction="sum",
        )
        
        # Normalize by number of positive samples
        box_loss = box_loss / max(1, sampled_pos_inds_subset.numel())
        
        return classification_loss, box_loss
    
    def extract_box_features(self, features, proposals):
        """
        Extract box features using RoIAlign with empty batch handling.
        """
        # Check if any proposals exist
        if len(proposals) == 0 or all(len(p) == 0 for p in proposals):
            # Return empty tensor with correct dimensions
            device = next(iter(features.values())).device
            return torch.zeros((0, self.box_feature_extractor.out_channels), device=device)
        
        # Convert dict of features to list
        features_list = [features[k] for k in sorted(features.keys())]
        
        # Apply RoI pooling to get fixed-size features
        box_features = self.box_roi_pool(features, proposals, [img.shape for img in features_list[:1]])
        
        # Process features through box feature extractor
        box_features = self.box_feature_extractor(box_features)
        
        return box_features

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        """
        Convert network outputs to bounding boxes and apply NMS.
        Handles batched inputs including empty batches.
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        
        # Handle empty inputs
        if sum(boxes_per_image) == 0:
            return [torch.zeros((0, 4), device=device) for _ in proposals], \
                [torch.zeros((0,), device=device) for _ in proposals], \
                [torch.zeros((0,), dtype=torch.int64, device=device) for _ in proposals]
                
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Process each image in the batch
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # Handle empty input for this image
            if boxes.shape[0] == 0:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros((0,), device=device))
                all_labels.append(torch.zeros((0,), dtype=torch.int64, device=device))
                continue
                
            boxes = self.clip_boxes_to_image(boxes, image_shape)
            
            # Create a final prediction per class
            image_boxes = []
            image_scores = []
            image_labels = []
            
            # Skip background class (class 0)
            for j in range(1, num_classes):
                class_boxes = boxes[:, j * 4:(j + 1) * 4]
                class_scores = scores[:, j]
                
                # Filter based on score threshold
                keep = class_scores > self.score_thresh
                
                # Handle case where no boxes pass threshold
                if keep.sum() == 0:
                    continue
                    
                class_boxes = class_boxes[keep]
                class_scores = class_scores[keep]
                
                # Apply NMS per class
                keep = nms(class_boxes, class_scores, self.nms_thresh)
                
                # Handle empty keep indices
                if keep.numel() == 0:
                    continue
                    
                # Keep top-k scoring boxes
                keep = keep[:self.detections_per_img]
                class_boxes = class_boxes[keep]
                class_scores = class_scores[keep]
                class_labels = torch.full((keep.size(0),), j, dtype=torch.int64, device=device)
                
                image_boxes.append(class_boxes)
                image_scores.append(class_scores)
                image_labels.append(class_labels)
            
            # Combine predictions across all classes
            if image_boxes:
                image_boxes = torch.cat(image_boxes, dim=0)
                image_scores = torch.cat(image_scores, dim=0)
                image_labels = torch.cat(image_labels, dim=0)
            else:
                # No detections for this image
                image_boxes = torch.zeros((0, 4), device=device)
                image_scores = torch.zeros((0,), device=device)
                image_labels = torch.zeros((0,), dtype=torch.int64, device=device)
            
            all_boxes.append(image_boxes)
            all_scores.append(image_scores)
            all_labels.append(image_labels)
        
        return all_boxes, all_scores, all_labels
    
    def assign_targets_to_proposals(self, proposals, targets):
        """
        Assign ground truth boxes and labels to each proposal.
        Handles batched inputs and empty batches with robust bounds checking.
        """
        matched_idxs = []
        labels = []
        regression_targets_list = []
        
        # Process each image in the batch
        for i, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
            gt_boxes = targets_per_image["boxes"]
            gt_labels = targets_per_image["labels"]
            
            # Handle empty gt boxes
            if len(gt_boxes) == 0:
                # No ground truth - all proposals are background
                device = proposals_per_image.device
                matches = torch.zeros(len(proposals_per_image), dtype=torch.int64, device=device) - 1  # All background
                proposal_labels = torch.zeros(len(proposals_per_image), dtype=torch.int64, device=device)
                
                # Sample proposals (all negative)
                sampled_inds = self.subsample_proposals_empty_gt(proposal_labels)
                
                # Create empty regression targets
                regression_targets = torch.zeros((len(sampled_inds), 4), device=device)
                
                matched_idxs.append(matches[sampled_inds])
                labels.append(proposal_labels[sampled_inds])
                regression_targets_list.append(regression_targets)
                proposals[i] = proposals_per_image[sampled_inds]
                continue
                
            # Regular processing with non-empty ground truth
            match_quality_matrix = box_iou(gt_boxes, proposals_per_image)
            
            # First, get standard matching from proposal → GT
            matched_vals, matches = match_quality_matrix.max(dim=0)
            
            # CRITICAL SAFETY CHECK: Ensure matches are within bounds of gt_labels
            # This prevents CUDA index out of bounds errors
            num_gt = len(gt_labels)
            matches = torch.clamp(matches, min=-1, max=num_gt-1)
            
            # Set labels based on IoU thresholds
            below_bg_thresh = matched_vals < self.bg_iou_thresh
            between_thresholds = (matched_vals >= self.bg_iou_thresh) & (matched_vals < self.fg_iou_thresh)
            
            matches[below_bg_thresh] = -1  # Background
            matches[between_thresholds] = -2  # Neutral, will be ignored
            
            # COCO-STYLE MATCHING: Ensure each GT box has at least one positive match
            # For each GT box, find its highest IoU with any proposal
            if len(proposals_per_image) > 0 and num_gt > 0:  # Check to avoid empty proposals or GT
                # Get best proposal for each GT box
                gt_to_proposal_ious, gt_to_proposal_idx = match_quality_matrix.max(dim=1)
                
                # For each GT, force-match the best proposal
                for gt_idx, proposal_idx in enumerate(gt_to_proposal_idx):
                    # Skip if IoU is too low (optional)
                    if gt_to_proposal_ious[gt_idx] < self.bg_iou_thresh:
                        continue
                        
                    # Force this proposal to match this GT
                    matches[proposal_idx] = gt_idx
            
            # Get labels for each proposal - SAFELY
            proposal_labels = torch.zeros_like(matches)
            pos_indices = matches >= 0  # Valid matches
            
            # Only assign labels to valid matches to avoid out-of-bounds indexing
            if pos_indices.sum() > 0:
                # First clamp again to be absolutely sure
                valid_matches = torch.clamp(matches[pos_indices], 0, num_gt-1)
                proposal_labels[pos_indices] = gt_labels[valid_matches]
            
            # Sample proposals for training (fixed batch size)
            sampled_inds = self.subsample_proposals(proposal_labels)
            
            # SAFETY CHECK: Ensure we have at least some valid proposals
            if len(sampled_inds) == 0:
                # Create empty tensors with the right shape
                device = proposals_per_image.device
                matched_idxs.append(torch.zeros(0, dtype=torch.int64, device=device))
                labels.append(torch.zeros(0, dtype=torch.int64, device=device))
                regression_targets_list.append(torch.zeros((0, 4), device=device))
                proposals[i] = proposals_per_image[0:0]  # Empty slice
                continue
                
            # Get matched GT boxes safely
            sampled_matches = matches[sampled_inds]
            valid_pos_indices = sampled_matches >= 0
            
            # Initialize with zeros
            matched_gt_boxes = torch.zeros((len(sampled_inds), 4), device=proposals_per_image.device)
            
            # Only assign matched boxes to positive indices
            if valid_pos_indices.sum() > 0:
                valid_matches = torch.clamp(sampled_matches[valid_pos_indices], 0, num_gt-1)
                matched_gt_boxes[valid_pos_indices] = gt_boxes[valid_matches]
                
            proposal_labels = proposal_labels[sampled_inds]
            
            # Compute regression targets
            regression_targets = self.box_coder.encode(
                matched_gt_boxes,
                proposals_per_image[sampled_inds]
            )
            
            matched_idxs.append(sampled_matches)
            labels.append(proposal_labels)
            regression_targets_list.append(regression_targets)
            proposals[i] = proposals_per_image[sampled_inds]
        

        # Combine results from all images
        if all(len(l) == 0 for l in labels):
            # All empty
            device = proposals[0].device
            return proposals, matched_idxs, torch.zeros(0, dtype=torch.int64, device=device), torch.zeros((0, 4), device=device)
        



        return proposals, matched_idxs, torch.cat(labels, dim=0), torch.cat(regression_targets_list, dim=0)

    def subsample_proposals_empty_gt(self, labels):
        """
        Sample proposals when there are no ground truths.
        All proposals will be negative/background.
        """
        # All labels are background (0)
        negative = torch.where(labels == 0)[0]
        
        # Sample up to batch_size_per_image negative examples
        num_neg = min(self.batch_size_per_image, negative.numel())
        
        # Randomly sample negative proposals
        perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        neg_idx = negative[perm]
        
        return neg_idx
    
    def compute_mask_loss(self, mask_logits, mask_targets, sampled_pos_inds_subset):
        """
        Compute mask loss using binary cross entropy with proper shape handling.
        """
        if sampled_pos_inds_subset.numel() == 0:
            # No positive samples, return zero loss
            return mask_logits.sum() * 0.0
        
        # Get selected mask logits
        selected_mask_logits = mask_logits[sampled_pos_inds_subset]
        
        # Check and align dimensions
        if len(selected_mask_logits.shape) == 4 and len(mask_targets.shape) == 3:
            # Reshape targets to match logits - expand targets for each class
            # This approach works if we need multi-class mask predictions
            mask_targets = mask_targets.unsqueeze(1)
            
            # If we have a multi-class setup and need to repeat targets for each class
            num_classes = selected_mask_logits.shape[1]
            if num_classes > 1:
                mask_targets = mask_targets.expand(-1, num_classes, -1, -1)
        
        mask_loss = F.binary_cross_entropy_with_logits(
            selected_mask_logits,
            mask_targets.to(dtype=torch.float32),
            reduction="mean",
        )
        
        return mask_loss
    
    def assign_masks_using_hungarian(self, proposals, targets):
        """
        Use Hungarian matching to assign proposals to ground truth masks.
        This ensures optimal 1-to-1 matching between proposals and GT instances.
        """

        
        matched_proposals = []
        matched_indices = []
        
        for proposals_per_img, targets_per_img in zip(proposals, targets):
            # Skip if no proposals or targets
            if len(proposals_per_img) == 0 or len(targets_per_img["boxes"]) == 0:
                matched_proposals.append(torch.zeros((0, 4), device=proposals_per_img.device))
                matched_indices.append(torch.zeros(0, dtype=torch.int64, device=proposals_per_img.device))
                continue
                
            # Compute cost matrix (IoU-based or other metrics)
            cost_matrix = -box_iou(proposals_per_img, targets_per_img["boxes"])
            
            # Apply Hungarian algorithm
            proposal_indices, target_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # Convert to torch tensors
            proposal_indices = torch.as_tensor(proposal_indices, device=proposals_per_img.device)
            target_indices = torch.as_tensor(target_indices, device=proposals_per_img.device)
            
            # Get matched proposals and their corresponding GT indices
            matched_proposals.append(proposals_per_img[proposal_indices])
            matched_indices.append(target_indices)
        
        return matched_proposals, matched_indices
    
    
    
    def clip_boxes_to_image(self, boxes, image_shape):
        """
        Clip boxes to be within image boundaries.
        """
        height, width = image_shape
        
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height)
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height)
        
        return boxes
    

class MSRGBInstanceModule(pl.LightningModule):
    """
    Mask R-CNN with Feature Pyramid Network
    """
    def __init__(self,
                 model_size='tiny',
                 rgb_in_channels=3,
                 ms_in_channels=5,
                 num_classes=2,
                 fusion_strategy='hierarchical',
                 fusion_type='attention',
                 drop_path_rate=0.0,
                 pretrained_backbone=None,
                 freeze_backbone=False,
                 lr = 1e-4,
                 weight_decay = 1e-4,
                 optimizer = "sgd",
                 momentum = 0.9,
                 # FPN parameters
                 fpn_out_channels=256,
                 # RPN parameters
                 rpn_anchor_sizes=((16,), (32,), (64,), (128,),),
                 rpn_aspect_ratios=((0.5, 1.0, 2.0),) * 4,
                 # RoI parameters
                 box_roi_output_size=7,
                 mask_roi_output_size=28,
                 sampling_ratio=2,
                 mask_threshold=0.5):
        
        super().__init__()


        self.lr = lr
        self.optimizer_name = optimizer
        self.weight_decay =weight_decay
        self.momentum = momentum
        self.pretrained_backbone = pretrained_backbone
        # Create backbone
        self.backbone = MSRGBConvNeXtFeatureExtractor(
            model_name=model_size,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            fusion_strategy=fusion_strategy,
            fusion_type=fusion_type,
            drop_path_rate=drop_path_rate
        )

        # Initialize from pretrained weights if provided
        if pretrained_backbone:
            self._load_pretrained_backbone(pretrained_backbone)

        if freeze_backbone:
            self.backbone.requires_grad_(False)
        
        # Get feature dimensions from backbone
        if 'flat' in self.backbone.feature_dims:
            # Exclude 'flat' key for FPN
            backbone_dims = {k: v for k, v in self.backbone.feature_dims.items() if k != 'flat'}
        else:
            backbone_dims = self.backbone.feature_dims
        print(backbone_dims)
            
        feature_dims = list(backbone_dims.values())
        feature_names = list(backbone_dims.keys())
        
        # Create Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=feature_dims,
            out_channels=fpn_out_channels
        )
        
        # Update backbone output channels for FPN
        self.backbone.out_channels = [fpn_out_channels] * len(feature_dims)
        
        # Create multi-level anchor generator
        anchor_generator = BatchedAnchorGenerator(
            sizes=rpn_anchor_sizes,
            aspect_ratios=rpn_aspect_ratios
        )
        
        # Create multi-scale RoI pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=feature_names,
            output_size=box_roi_output_size,
            sampling_ratio=sampling_ratio
        )
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=feature_names,
            output_size=mask_roi_output_size,
            sampling_ratio=sampling_ratio
        )
        
        # Create multi-level RPN
        rpn_head = MultiLevelRPNHead(
            in_channels=fpn_out_channels,
            num_anchors_per_location=anchor_generator.num_anchors_per_location()[0]
        )
        self.box_coder = BatchedBoxCoder(weights=(1.0,1.0,1.0,1.0))
        
        self.rpn = BatchedRegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.5,
            box_coder = self.box_coder
        )
        
        # Create box predictor
        box_predictor = FastRCNNPredictor(
            1024,
            num_classes
        )
        
                # Create improved mask heads
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = ImprovedMaskHead(
            fpn_out_channels,  # FPN output channels
            mask_layers, 
            mask_dilation
        )
        
        mask_predictor_in_channels = mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = ImprovedMaskPredictor(
            mask_predictor_in_channels, 
            mask_dim_reduced, 
            num_classes
        )
        
        # Create improved RoI heads
        self.roi_heads = ImprovedRoIHeads(
            box_roi_pool=box_roi_pool,
            box_feature_extractor=self._create_box_head(fpn_out_channels, box_roi_output_size),
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,  # 0.3?
            batch_size_per_image=512,
            positive_fraction=0.5,  # Balanced sampling
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,  # More aggressive NMS
            detections_per_img=100,
            mask_roi_pool=mask_roi_pool,
            mask_feature_extractor=mask_head,
            mask_predictor=mask_predictor,
            box_coder= self.box_coder
        )
        # Store threshold for mask post-processing
        self.mask_threshold = mask_threshold

    def _create_box_head(self, in_channels, roi_output_size):
        """Create the box head for feature extraction before classification"""
        input_size = in_channels * roi_output_size * roi_output_size
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )


    def _post_process_masks(self, detections):
        """Apply mask threshold to predictions"""
        for detection in detections:
            if 'masks' in detection:
                masks = detection['masks']
                detection['masks'] = (masks > self.mask_threshold).to(dtype=torch.uint8)
    
    def forward(self, rgb=None, ms=None, targets=None):
        """
        Forward pass through FPN-based Mask R-CNN
        
        Args:
            rgb: RGB images (B, 3, H, W)
            ms: Multispectral images (B, 5, H, W)
            targets: Training targets (list of dicts)
        
        Returns:
            During training: Dictionary of losses
            During inference: List of predictions
        """
        # Extract features from backbone
        backbone_features = self.backbone(rgb=rgb, ms=ms)
        spatial_features = {}
        for name, feature in backbone_features.items():
            if len(feature.shape) == 4:  # Only keep 4D feature maps (B, C, H, W)
                spatial_features[name] = feature

        # Apply Feature Pyramid Network
        fpn_features = self.fpn(spatial_features)
        
        # Get image sizes for RPN, rgb, or ms is only used for shape (B,C,H,W)
        if rgb is not None:
            image_sizes = [img.shape[-2:] for img in rgb]
            # Run RPN on FPN features
            proposals, rpn_losses = self.rpn(rgb, fpn_features, targets)
        elif ms is not None:
            image_sizes = [img.shape[-2:] for img in ms]
            # Run RPN on FPN features
            proposals, rpn_losses = self.rpn(ms, fpn_features, targets)
        else:
            raise ValueError("Either rgb or ms input must be provided")
            
        # Run RoI heads on FPN features
        detections, roi_losses = self.roi_heads(fpn_features, proposals, image_sizes, targets)

        if self.training:
            # Combine losses during training
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses
        else:
            # Apply mask threshold during inference
            self._post_process_masks(detections)
            return detections

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """
        batch format:
            'rgb': Tensor[B, C1, H, W]
            'ms' : Tensor[B, C2, H, W]
            'boxes': List[List[Tensor(5,)]] each tensor = [cls, x, y, w, h], normalized
            'instance_masks': List[List[Tensor(H, W)]]
        """
        rgb_batch = batch.get('rgb')
        ms_batch = batch.get('ms')
        if rgb_batch is not None:
            batch_size,c,hb,wb, = rgb_batch.shape
            device = rgb_batch.device
        elif ms_batch is not None:
            batch_size,c,hb,wb = ms_batch.shape
            device = ms_batch.device
        boxes = batch.get("boxes")
        instance_masks  = batch.get("instance_masks")

        # prepare inputs
        targets = []
        for i in range(batch_size):
            boxes_i = boxes[i]
            masks_i = instance_masks[i]
            if len(boxes_i) == 0:
                # no objects
                targets.append({'boxes': torch.zeros((0,4), dtype=torch.float32, device =device ),
                                'labels': torch.zeros((0,), dtype=torch.int64, device =device ),
                                'masks': torch.zeros((0, int(hb), int(wb)), dtype=torch.uint8, device =device)})
                continue
            # convert each [cls, x_c, y_c, w, h] to xyxy in pixels
            H, W = int(hb), int(wb)
            xyxy = []
            labels = []
            for b in boxes_i:
                cls, xc, yc, w, h = b
                x1 = (xc - w/2) * W
                y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W
                y2 = (yc + h/2) * H
                xyxy.append(torch.stack([x1, y1, x2, y2]))
                labels.append(cls.to(torch.int64))
                
            target = {
                'boxes': torch.stack(xyxy),
                'labels': torch.stack(labels),
                'masks': torch.stack(masks_i).to(torch.uint8)
            }
            targets.append(target)
   
        # forward and compute losses
        loss_dict = self.forward(ms=ms_batch,rgb=rgb_batch, targets=targets)

        loss = sum(loss for loss in loss_dict.values())
        # log losses
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        # same processing as training_step but logs under 'val/'
        rgb_batch = batch.get('rgb')
        ms_batch = batch.get('ms')
        if rgb_batch is not None:
            batch_size = rgb_batch.shape[0]
        elif ms_batch is not None:
            batch_size = ms_batch.shape[0]
    
        targets = []
        for i in range(batch_size):

            boxes_i = batch['boxes'][i]
            masks_i = batch['instance_masks'][i]
            if len(boxes_i) == 0:
                targets.append({'boxes': torch.zeros((0,4), dtype=torch.float32),
                                'labels': torch.zeros((0,), dtype=torch.int64),
                                'masks': torch.zeros((0, rgb_batch.shape[2], rgb_batch.shape[3]), dtype=torch.uint8)})
                continue
            H, W = rgb_batch.shape[2], rgb_batch.shape[3]
            xyxy = []
            labels = []
            for b in boxes_i:
                cls, xc, yc, w, h = b
                x1 = (xc - w/2) * W
                y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W
                y2 = (yc + h/2) * H
                xyxy.append(torch.stack([x1, y1, x2, y2]))
                labels.append(cls.to(torch.int64))
            targets.append({
                'boxes': torch.stack(xyxy),
                'labels': torch.stack(labels),
                'masks': torch.stack(masks_i).to(torch.uint8)
            })

        loss_dict = self.forward(rgb = rgb_batch, ms = ms_batch, targets = targets)
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/total_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.optimizer_name.upper() == 'SGD':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.upper() == 'ADAM':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

          
    def _load_pretrained_backbone(self, checkpoint_path):
        """Load weights from a PyTorch Lightning checkpoint file"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict - handle both direct state_dict and Lightning format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'backbone.' prefix if it exists (common in Lightning models)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]
                    new_state_dict[new_key] = value
                # Also handle 'feature_extractor.' prefix
                elif key.startswith('feature_extractor.'):
                    new_key = key[len('feature_extractor.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        
        # Load weights to backbone
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

    def predict_masks_and_boxes(self, rgb=None, ms=None, confidence_threshold=0.5):
        """
        Generate predictions for visualization without requiring ground truth labels.
        
        Args:
            rgb: RGB images tensor of shape (B, 3, H, W) or None
            ms: Multispectral images tensor of shape (B, 5, H, W) or None
            confidence_threshold: Minimum confidence score to keep a detection
            
        Returns:
            List of dictionaries, one per image, each containing:
                - 'boxes': tensor of shape (N, 4) with [x1, y1, x2, y2] coordinates
                - 'masks': binary tensor of shape (N, H, W)
                - 'scores': confidence scores tensor of shape (N,)
                - 'labels': class labels tensor of shape (N,)
        """
        # Set model to evaluation mode
        original_mode = self.training
        self.eval()
        
        # Ensure input is on the right device
        device = next(self.parameters()).device
        if rgb is not None and rgb.device != device:
            rgb = rgb.to(device)
        if ms is not None and ms.device != device:
            ms = ms.to(device)
        
        # Disable gradient computation for prediction
        with torch.no_grad():
            # Extract features from backbone
            backbone_features = self.backbone(rgb=rgb, ms=ms)
            
            # Extract spatial features (4D tensors)
            spatial_features = {}
            for name, feature in backbone_features.items():
                if len(feature.shape) == 4:  # Only keep 4D feature maps
                    spatial_features[name] = feature
            
            # Apply Feature Pyramid Network
            fpn_features = self.fpn(spatial_features)
            
            # Get image sizes
            if rgb is not None:
                images = rgb
                image_sizes = [img.shape[-2:] for img in rgb]
            else:
                images = ms
                image_sizes = [img.shape[-2:] for img in ms]
            
            # Run RPN to get proposals
            proposals, _ = self.rpn(images, fpn_features, None)
            
            # Get base device for empty tensor creation
            base_device = next(iter(fpn_features.values())).device
            
            # Run classification and regression without mask prediction
            # (to avoid the KeyError with features["0"])
            box_features = self.roi_heads.extract_box_features(fpn_features, proposals)
            
            # Direct inference using box features
            if box_features.numel() > 0:
                class_logits, box_regression = self.roi_heads.box_predictor(box_features)
                boxes, scores, labels = self.roi_heads.postprocess_detections(
                    class_logits, box_regression, proposals, image_sizes
                )
                
                # Build detection results
                results = []
                for i in range(len(boxes)):
                    result_dict = {
                        'boxes': boxes[i],
                        'scores': scores[i],
                        'labels': labels[i],
                    }
                    
                    # Filter by confidence
                    if len(scores[i]) > 0:
                        keep = scores[i] >= confidence_threshold
                        result_dict = {k: v[keep] for k, v in result_dict.items()}
                    
                    # Add to results
                    results.append(result_dict)
            else:
                # Handle empty case
                results = [
                    {
                        'boxes': torch.zeros((0, 4), device=base_device),
                        'scores': torch.zeros((0,), device=base_device),
                        'labels': torch.zeros((0,), dtype=torch.int64, device=base_device)
                    }
                    for _ in range(len(proposals))
                ]
            
            # Add empty masks for now (we'll compute them in a separate step)
            for i, result in enumerate(results):
                if rgb is not None:
                    h, w = rgb.shape[2:4]
                else:
                    h, w = ms.shape[2:4]
                result['masks'] = torch.zeros(
                    (len(result['boxes']), h, w), 
                    dtype=torch.uint8, 
                    device=result['boxes'].device
                )
            
            # If we have detections, compute masks for them
            for i, result in enumerate(results):
                if len(result['boxes']) > 0:
                    # Prepare boxes for mask RoI pooling
                    mask_boxes = [result['boxes']]
                    
                    # Only try to compute masks if we have a mask predictor
                    if hasattr(self.roi_heads, 'mask_predictor') and self.roi_heads.mask_predictor is not None:
                        try:
                            # Manually extract mask features to avoid KeyError
                            features_list = list(fpn_features.values())
                            # Get mask features
                            mask_features = self.roi_heads.mask_roi_pool(
                                fpn_features, 
                                mask_boxes, 
                                [img.shape for img in features_list[:1]]
                            )
                            mask_features = self.roi_heads.mask_feature_extractor(mask_features)
                            
                            # Predict masks
                            mask_logits = self.roi_heads.mask_predictor(mask_features)
                            
                            # Convert logits to binary masks
                            mask_probs = mask_logits.sigmoid()
                            
                            # Get binary masks
                            if len(mask_probs.shape) == 4 and mask_probs.shape[1] > 1:
                                # Multi-class case: select channel based on predicted class
                                mask_probs = mask_probs[torch.arange(mask_probs.shape[0]), result['labels']]
                            elif len(mask_probs.shape) == 4:
                                # Single class case (foreground/background)
                                mask_probs = mask_probs[:, 0]
                                
                            # Apply threshold and resize masks to image size
                            binary_masks = (mask_probs > self.mask_threshold).float()
                            
                            # Resize masks to original image size
                            resized_masks = []
                            for mask, box in zip(binary_masks, result['boxes']):
                                # Convert box to integers for ROI
                                x1, y1, x2, y2 = map(int, box.tolist())
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                # Skip invalid boxes
                                if x2 <= x1 or y2 <= y1:
                                    resized_masks.append(torch.zeros((h, w), device=base_device))
                                    continue
                                    
                                # Create full-size mask
                                full_mask = torch.zeros((h, w), device=base_device)
                                
                                # Resize the mask to the box size
                                box_h, box_w = y2 - y1, x2 - x1
                                if box_h > 0 and box_w > 0:  # Valid box dimensions
                                    # Resize the mask to the box dimensions
                                    resized = F.interpolate(
                                        mask.unsqueeze(0).unsqueeze(0),
                                        size=(box_h, box_w),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze()
                                    
                                    # Place the resized mask in the full image
                                    full_mask[y1:y2, x1:x2] = resized
                                    
                                resized_masks.append(full_mask)
                                
                            # Stack masks
                            if resized_masks:
                                result['masks'] = torch.stack(resized_masks).to(torch.uint8)
                        except Exception as e:
                            print(f"Error computing masks: {e}")
                            # Keep the empty masks on error
        
        # Restore original training mode
        self.train(original_mode)
        
        return results