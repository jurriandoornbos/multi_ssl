# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from .fastsiam import FastSiam, DualObjectiveFastSiam

def build_model(args):

    if args.ssl_method == "fastsiam":
        model = FastSiam(
            backbone=args.backbone,
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            pred_dim=args.pred_dim,
            lr=args.lr,
            in_channels=args.in_channels,
            batch_size = args.batch_size,
            epochs = args.epochs,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dataset_size = args.dataset_size)

    elif args.ssl_method == "galileo-fastsiam":
        model = DualObjectiveFastSiam(
            backbone=args.backbone,
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            pred_dim=args.pred_dim,
            lr=args.lr,
            in_channels=args.in_channels,
            batch_size = args.batch_size,
            epochs = args.epochs,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    
    return model
        