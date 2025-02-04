# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingScheduler(_LRScheduler):
    """
    Custom LR scheduler that:
      - Warms up in the first half of the first epoch (linear increase).
      - Then applies cosine annealing for the remainder of training.
    """
    def __init__(self,  optimizer, total_steps, steps_per_epoch, 
                 warmup_fraction=0.5, lr_min=1e-7, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of training steps (epochs * steps_per_epoch).
            steps_per_epoch (int): Number of steps (batches) in one epoch.
            warmup_fraction (float): Fraction of the first epoch used for warmup. 
                                     0.5 means the first half of the first epoch.
            eta_min (float): Minimum learning rate.
            last_epoch (int): The index of the last epoch. 
                              (By PyTorch convention, set this to -1 when initializing.)
        """
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        # Number of steps to warm up. For half of the first epoch:
        self.warmup_steps = int(warmup_fraction * self.steps_per_epoch)
        self.lr_min = lr_min

        super(WarmupCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # `last_epoch` in _LRScheduler counts how many times step() has been called.
        current_step = self.last_epoch

        # If warmup has not finished, linearly ramp up from 0 -> base_lr
        if current_step <= self.warmup_steps:
            warmup_ratio = current_step / float(self.warmup_steps)
            return [base_lr * warmup_ratio for base_lr in self.base_lrs]
        
        # If we've passed all training steps, just return minimum LR (or you can clamp)
        if current_step >= self.total_steps:
            return [self.lr_min for _ in self.base_lrs]
        
        # Otherwise, apply cosine annealing
        # progress goes from 0 (end of warmup) to 1 (end of training)
        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [
            self.lr_min + (base_lr - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
    
    def step(self, global_step=None):
        """
        Overrides the default step() to allow manually setting the current "step"
        (e.g., from PyTorch Lightning's trainer.global_step).
        
        If `global_step` is provided, we set `self.last_epoch` to (global_step - 1)
        so that `current_step = last_epoch + 1` matches `global_step`.
        """
        if global_step is not None:
            # We want get_lr() to see 'current_step' == global_step, 
            # so we store global_step - 1 in self.last_epoch.
            self.last_epoch = global_step - 1
        super().step()