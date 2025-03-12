# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import math
import torch
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


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts learning rate scheduler.
    
    This implements the learning rate schedule described in the paper
    "SGDR: Stochastic Gradient Descent with Warm Restarts"
    (https://arxiv.org/abs/1608.03983)
    
    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart cycle.
        T_mult (int, optional): A factor to increase T_i after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        warmup_epochs (int, optional): Number of epochs for linear warmup. Default: 0.
        warmup_start_lr (float, optional): Initial learning rate for warmup. Default: 0.
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, 
                 warmup_epochs=0, warmup_start_lr=0):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.cycle = 0
        
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize optimizer base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def get_lr(self):
        # Handle warmup phase
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        
        # Past warmup phase
        epoch_adjusted = self.last_epoch - self.warmup_epochs
        
        # Check if we need to restart
        if epoch_adjusted >= self.T_i:
            # Reset T_cur and increase T_i for the next cycle
            self.T_cur = epoch_adjusted - self.T_i
            self.T_i = self.T_i * self.T_mult
            self.cycle += 1
        else:
            self.T_cur = epoch_adjusted
        
        # Calculate LR using cosine annealing formula
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]


class CosineAnnealingWarmRestartsDecay(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts and learning rate decay between cycles.
    
    This extends the SGDR approach by adding a decay factor between restarts,
    which can help prevent overfitting in later cycles.
    
    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart cycle.
        T_mult (int, optional): A factor to increase T_i after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        warmup_epochs (int, optional): Number of epochs for linear warmup. Default: 0.
        warmup_start_lr (float, optional): Initial learning rate for warmup. Default: 0.
        decay_factor (float, optional): Factor to decay max learning rate each cycle. Default: 0.8.
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, 
                 warmup_epochs=0, warmup_start_lr=0, decay_factor=0.8):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.decay_factor = decay_factor
        self.cycle = 0
        
        super(CosineAnnealingWarmRestartsDecay, self).__init__(optimizer, last_epoch)
        
        # Initialize optimizer base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_max_lrs = list(self.base_lrs)
    
    def get_lr(self):
        # Handle warmup phase
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        
        # Past warmup phase
        epoch_adjusted = self.last_epoch - self.warmup_epochs
        
        # Check if we need to restart
        if epoch_adjusted >= self.T_i:
            # Reset T_cur and increase T_i for the next cycle
            self.T_cur = epoch_adjusted - self.T_i
            self.T_i = self.T_i * self.T_mult
            self.cycle += 1
            
            # Decay the maximum learning rate for the next cycle
            self.current_max_lrs = [lr * self.decay_factor for lr in self.current_max_lrs]
        else:
            self.T_cur = epoch_adjusted
        
        # Calculate LR using cosine annealing formula with decayed max lr
        return [self.eta_min + (max_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for max_lr in self.current_max_lrs]


class OneCycleScheduler(_LRScheduler):
    """
    One Cycle Learning Rate Scheduler.
    
    Implementation of the OneCycle policy from the paper:
    "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
    (https://arxiv.org/abs/1708.07120)
    
    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        total_epochs (int): Total number of epochs for training.
        max_lr (float or list): Maximum learning rate. If list, one per param group.
        pct_start (float): Percentage of training spent increasing the learning rate. Default: 0.3.
        div_factor (float): Initial learning rate is max_lr/div_factor. Default: 25.
        final_div_factor (float): Final learning rate is max_lr/final_div_factor. Default: 10000.
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    
    def __init__(self, optimizer, total_epochs, max_lr, pct_start=0.3, 
                 div_factor=25.0, final_div_factor=10000.0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        
        # Convert max_lr to list if it's not already
        if not isinstance(max_lr, list) and not isinstance(max_lr, tuple):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = list(max_lr)
            
        # Calculate initial and final learning rates
        self.initial_lrs = [max_lr / self.div_factor for max_lr in self.max_lrs]
        self.final_lrs = [max_lr / self.final_div_factor for max_lr in self.max_lrs]
        
        # Calculate step sizes
        self.step_size_up = int(total_epochs * self.pct_start)
        self.step_size_down = total_epochs - self.step_size_up
        
        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.step_size_up:
            # We're in the ramp-up phase
            return [initial_lr + (max_lr - initial_lr) * (self.last_epoch / self.step_size_up)
                    for initial_lr, max_lr in zip(self.initial_lrs, self.max_lrs)]
        else:
            # We're in the annealing phase
            current_step = self.last_epoch - self.step_size_up
            return [max_lr + (final_lr - max_lr) * (current_step / self.step_size_down)
                    for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)]