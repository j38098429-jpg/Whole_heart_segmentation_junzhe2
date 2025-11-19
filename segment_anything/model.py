from segment_anything import sam_model_registry

    
def build_model(args, device):
    model = sam_model_registry[args.vit_type](checkpoint=args.resume,)

    return model.to(device=device)
 
