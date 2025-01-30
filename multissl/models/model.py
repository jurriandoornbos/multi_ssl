from .fastsiam import FastSiam

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
            weight_decay=args.weight_decay,)
        return model