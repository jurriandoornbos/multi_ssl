import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, nms, batched_nms
import torch.optim as optim
from typing import Dict, Any, Tuple, List, Optional
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor

import math
from .msrgb_convnext import MSRGBConvNeXtFeatureExtractor

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
        device: torch.device = torch.device("cpu"),
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
    """
    Box coder that works with batched tensors instead of lists
    """
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        self.weights = weights
    
    def encode(self, reference_boxes, proposals):
        """
        Encode bounding box regression targets
        """
        if isinstance(reference_boxes, list):
            reference_boxes = torch.cat(reference_boxes, dim=0)
        if isinstance(proposals, list):
            proposals = torch.cat(proposals, dim=0)
            
        # Box centers
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        
        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        
        # Prevent division by zero
        ex_widths = torch.clamp(ex_widths, min=1e-6)
        ex_heights = torch.clamp(ex_heights, min=1e-6)
        
        # Compute targets
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
        
        # Apply weights
        wx, wy, ww, wh = self.weights
        targets = torch.stack((targets_dx * wx, targets_dy * wy, 
                             targets_dw * ww, targets_dh * wh), dim=1)
        
        return targets
    
    def decode(self, rel_codes, boxes):
        """
        Decode bounding box predictions
        """
        # Handle both single tensor and list inputs
        if isinstance(boxes, list):
            boxes = torch.cat(boxes, dim=0)
            
        # Box centers and sizes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        # Apply weights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0] / wx
        dy = rel_codes[:, 1] / wy
        dw = rel_codes[:, 2] / ww
        dh = rel_codes[:, 3] / wh
        
        # Prevent sending too large values to exp()
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))
        
        # Decode
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        # Convert to (x1, y1, x2, y2) format
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
        
        return pred_boxes


class BatchedRegionProposalNetwork(nn.Module):
    """
    Region Proposal Network that works with batched tensor inputs
    """
    def __init__(
        self,
        anchor_generator,
        head,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_top_n: Dict[str, int] = None,
        post_nms_top_n: Dict[str, int] = None,
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
        device: str = "cuda"
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BatchedBoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        
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
        """Compute RPN losses"""
        # Classification loss
        sampled_pos_inds, sampled_neg_inds = self.subsample_labels(labels)
        sampled_inds = torch.where(torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0))[0]
        
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds].float()
        )
        
        # Regression loss (only for positive samples)
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())
        
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
        """Filter proposals using NMS"""
        num_images = proposals.shape[0]
        device = proposals.device
        
        # Get scores and apply score threshold
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) 
                 for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, dim=0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        
        # Select top-k proposals before NMS
        top_n = self.pre_nms_top_n["training" if self.training else "testing"]
        top_n = min(top_n, objectness.shape[1])
        
        objectness, top_n_idx = objectness.topk(top_n, dim=1, sorted=True)
        
        batch_idx = torch.arange(num_images, device=device)[:, None]
        proposals = proposals[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        
        final_boxes = []
        final_scores = []
        
        # Process each image separately
        for boxes, scores, img_shape, lvl in zip(proposals, objectness, image_shapes, levels):
            # Clip boxes to image boundaries
            # After (out-of-place, safe):
            boxes = boxes.clone()  # Ensure we have a non-view tensor
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=img_shape[1])
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=img_shape[0])
            
            # Remove small boxes
            keep = self.remove_small_boxes(boxes, min_size=1e-3)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            
            # NMS
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)
            
            # Select post-NMS top-k
            post_nms_top_n = self.post_nms_top_n["training" if self.training else "testing"]
            keep = keep[:post_nms_top_n]
            
            final_boxes.append(boxes[keep])
            final_scores.append(scores[keep])
            
        return final_boxes, final_scores
    
    def remove_small_boxes(self, boxes, min_size):
        """Remove boxes with width or height less than min_size"""
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        return torch.where(keep)[0]
    
    def assign_targets_to_anchors(self, anchors, targets):
        """Assign ground truth targets to anchors for training"""
        # This is a simplified version - you may need more sophisticated assignment
        # based on your specific requirements
        
        labels = []
        matched_gt_boxes = []
        
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if self.training and targets_per_image is not None:
                # Compute IoU between anchors and ground truth boxes
                gt_boxes = targets_per_image["boxes"]
                iou_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                
                # Simple assignment based on max IoU
                matched_vals, matched_idxs = iou_matrix.max(dim=0)
                
                # Create labels
                labels_per_image = torch.zeros_like(matched_vals, dtype=torch.long)
                labels_per_image[matched_vals >= self.fg_iou_thresh] = 1
                labels_per_image[matched_vals < self.bg_iou_thresh] = 0
                
                matched_gt_boxes_per_image = gt_boxes[matched_idxs]
            else:
                # During inference, no targets
                labels_per_image = torch.zeros(anchors_per_image.shape[0], dtype=torch.long, device=anchors_per_image.device)
                matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image)
            
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            
        return labels, matched_gt_boxes
    
    def box_similarity(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        iou = inter / (area1[:, None] + area2 - inter)
        return iou
    
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

        # Generate anchors
        anchors = self.anchor_generator(images, feature_maps, device = self.device)
        
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
        proposals = self.box_coder.decode(pred_bbox_deltas.view(-1, 4), expanded_anchors)
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
            
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                
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
        
        # Initialize weights
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
            
        # Special initialization for classification head
        nn.init.constant_(self.cls_logits.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)))
        
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
            
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                
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
        
        # Initialize weights
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
            
        # Special initialization for classification head
        nn.init.constant_(self.cls_logits.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)))
        
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
                 optimizer = "adam",
                 # FPN parameters
                 fpn_out_channels=256,
                 # RPN parameters
                 rpn_anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
                 rpn_aspect_ratios=((0.5, 1.0, 2.0),) * 5,
                 # RoI parameters
                 box_roi_output_size=7,
                 mask_roi_output_size=14,
                 sampling_ratio=2,
                 mask_threshold=0.5):
        
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer
        self.weight_decay =weight_decay
        
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
        
        self.rpn = BatchedRegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7,
        )
        
        # Create box predictor
        box_predictor = FastRCNNPredictor(
            fpn_out_channels * box_roi_output_size * box_roi_output_size,
            num_classes
        )
        
        # Create mask heads
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(
            fpn_out_channels,  # FPN output channels
            mask_layers, 
            mask_dilation
        )
        
        mask_predictor_in_channels = mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(
            mask_predictor_in_channels, 
            mask_dim_reduced, 
            num_classes
        )
        
        # Create RoI heads
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=self._create_box_head(fpn_out_channels, box_roi_output_size),
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
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
    
        
        # Get image sizes for RPN
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
            batch_size = rgb_batch.shape[0]
        elif ms_batch is not None:
            batch_size = ms_batch.shape[0]

        # prepare inputs
        targets = []
        for i in range(batch_size):
            boxes_i = batch['boxes'][i]
            masks_i = batch['instance_masks'][i]
            if len(boxes_i) == 0:
                # no objects
                targets.append({'boxes': torch.zeros((0,4), dtype=torch.float32),
                                'labels': torch.zeros((0,), dtype=torch.int64),
                                'masks': torch.zeros((0, rgb_batch.shape[2], rgb_batch.shape[3]), dtype=torch.uint8)})
                continue
            # convert each [cls, x_c, y_c, w, h] to xyxy in pixels
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
    
        images = []
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

        loss_dict = self.forward(images, targets)
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

