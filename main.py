import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import torch
import torch.nn as nn
import torch.distributed as dist
from timm.models.vision_transformer import VisionTransformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time

from model.dpt_head import create_dpt_head

def get_encoder():
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

def get_global_transformer(encoder):       
    encoder_layer = TransformerEncoderLayer(
        d_model=encoder.embed_dim,  # Match ViT-Base dimension
        nhead=12,
        dim_feedforward=3072,
        dropout=0.0,
        activation='gelu',
    )

    transformer = TransformerEncoder(
        encoder_layer,
        num_layers=6,
    )
    return transformer

def get_decoder(encoder):
    def map_dinov2_to_croco_naming(net, depth_mode="square", conf_mode="exp"):
        net.enc_embed_dim = net.embed_dim
        net.dec_embed_dim = net.embed_dim
        net.dec_depth = net.n_blocks - 1
        net.depth_mode = depth_mode
        net.conf_mode = conf_mode
        return net

    dpt_head = create_dpt_head(
        map_dinov2_to_croco_naming(encoder),
    )
    dpt_head.postprocess = None
    return dpt_head

class ParallelFast3r(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_frames=8):
        super().__init__()
        
        # Components
        self.encoder = get_encoder()
        self.transformer = get_global_transformer(self.encoder)
        self.decoder = get_decoder(self.encoder)

        self.num_frames = num_frames
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    @torch.inference_mode()
    # @torch.compile()
    # @torch.jit.script
    def forward(self, images):
        # Ensure input is on the correct device
        # images = images.to(torch.cuda.current_device())
        B, K, C, H, W = images.shape
        #   K, C, H, W = images.shape
        
        # Split frames across GPUs
        assert K % self.world_size == 0, "Number of frames must be divisible by number of GPUs"
        split_size = K // self.world_size
        images_split = torch.chunk(images, self.world_size, dim=1)
        local_images = images_split[self.rank].contiguous()

        # Reshape for ViT: (B*K_local, C, H, W)
        local_images = local_images.view(-1, C, H, W)
        
        # Encode with ViT
        # local_tokens = self.encoder.forward_features(local_images)  # Shape: (B*K_local, N, C)
        local_tokens_multilayer = self.encoder.get_intermediate_layers(
            local_images, 
            n=self.encoder.n_blocks
        )  # Shape: (B*K_local, N, C)
        local_tokens = local_tokens_multilayer[-1]
        
        # run self.transformer
        # We could shard this along the batch dimensions, but for now we jsut recompute this on each GPU
        
        # Gather tokens to (B, K*N), run transformer, and scatter
        gathered_tokens = [torch.zeros_like(local_tokens) for _ in range(self.world_size)]
        dist.all_gather(gathered_tokens, local_tokens.contiguous())
        tokens = torch.cat(gathered_tokens, dim=0)  # Shape: (B*K, N, C)
        tokens = tokens.view(B, -1, self.encoder.embed_dim) # Shape: (B, K*N, C)
    
        transformed = self.transformer(tokens.contiguous())
    
        transformed = transformed.view(B * K, -1, self.encoder.embed_dim) # Shape: (B*K, N, C)
        transformed_split = torch.chunk(transformed, self.world_size, dim=0)
        local_transformed = transformed_split[self.rank]
        
        # Run decoder
        tokens_multilayer = [tok for tok in local_tokens_multilayer]
        tokens_multilayer[-1] = local_transformed
        local_decoded = self.decoder(tokens_multilayer, img_info=(H, W))
        return local_decoded
        # Gather final outputs
        gathered_outputs = [torch.zeros_like(local_decoded) for _ in range(self.world_size)]
        dist.all_gather(gathered_outputs, local_decoded.contiguous())
        output = torch.cat(gathered_outputs, dim=0)
        
        # Reshape back to (B, K, C, H, W)
        output = output.view(B, K, C, H, W)
        # output = output.view(K, C, H, W)
         
        return output


def run_model(
        rank: int, 
        world_size: int,
        batch_size: int,
        image_size: int, 
        patch_size: int, 
        num_frames: int,
        num_warmup: int = 3,
        num_trials: int = 10,
        dtype: str = 'half'
    ) -> None:
    torch.set_grad_enabled(False)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    model = ParallelFast3r(
        image_size=image_size,
        patch_size=patch_size,
        num_frames=num_frames
    ).to(rank).eval()
    
    # Print model parameters
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Create dummy input
    images = torch.randn(batch_size, num_frames, 3, image_size, image_size).to(rank)
    
    # Warmup runs with autocast
    if rank == 0:
        print(f"\nPerforming {num_warmup} warmup runs...")
    for _ in range(num_warmup):
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
            _ = model(images)
    torch.cuda.synchronize()
    
    # Multiple timing trials with autocast
    if rank == 0:
        print(f"\nRunning {num_trials} timing trials...")
    
    times = []
    peak_memories = []
    
    for trial in range(num_trials):
        torch.cuda.reset_peak_memory_stats(rank)
        start_time = time.time()
        
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
            output = model(images)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        trial_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(rank) / (1024 ** 3)  # Convert to GB
        
        times.append(trial_time)
        peak_memories.append(peak_memory)
        
        if rank == 0:
            print(f"Trial {trial+1}: {trial_time:.3f}s, {peak_memory:.2f}MB")
    
    if rank == 0:
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        avg_memory = sum(peak_memories) / len(peak_memories)
        
        print("\nResults:")
        print(f"Output shape: {output.shape}")
        print(f"Average forward pass time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Average peak GPU memory usage: {avg_memory:.2f}GB")

    dist.destroy_process_group()

# Run this on an 8 GPU machine using python main.py --batch_size 2 --image_size 224 --patch_size 16 --num_frames 16
if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    from functools import partial
    import math
    import os
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can use any free port number

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--image_size', type=int, default=224, help='Size of input images')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of image patches')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to process')
    parser.add_argument('--num_warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of timing trials')
    parser.add_argument('--dtype', type=str, default='half', help='Data type for processing')
    parser.add_argument('--sdpa', type=str, default='math', help='SDPA type for processing')
    args = parser.parse_args()

    batch_size = args.batch_size
    image_size = args.image_size 
    patch_size = args.patch_size
    num_frames = args.num_frames

    world_size = torch.cuda.device_count()
    if args.dtype == 'half':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(f"Running with dtype: {dtype}")

    if args.sdpa == 'math':
        torch.backends.cuda.enable_math_sdp(True)
    elif args.sdpa == 'flash':
        torch.backends.cuda.enable_flash_sdp(True)
    elif args.sdpa == 'mem':
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    else:
        raise ValueError(f"Invalid SDPA type: {args.sdpa}")

    mp.spawn(
        run_model,
        args=(
            world_size, 
            args.batch_size, 
            args.image_size, 
            args.patch_size, 
            args.num_frames, 
            args.num_warmup, 
            args.num_trials,
            dtype
        ),
        nprocs=world_size,
        join=True)
