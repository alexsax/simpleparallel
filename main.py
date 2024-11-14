import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import torch
import torch.nn as nn
import torch.distributed as dist
from timm.models.vision_transformer import VisionTransformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time

from model.dpt_head import create_dpt_head
from torch.nn.attention import SDPBackend, sdpa_kernel

SDPA_TYPES = {
    'flash': SDPBackend.FLASH_ATTENTION,
    'mem': SDPBackend.EFFICIENT_ATTENTION,
    'math': SDPBackend.MATH,
}

def get_encoder():
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

def get_global_transformer(encoder, sdpa):
    torch.backends.cuda.enable_mem_efficient_sdp(sdpa == SDPBackend.EFFICIENT_ATTENTION)
    torch.backends.cuda.enable_flash_sdp(sdpa == SDPBackend.FLASH_ATTENTION)
    torch.backends.cuda.enable_math_sdp(sdpa == SDPBackend.MATH)
    from model.global_transformer import Fast3RDecoder
    # transformer = Fast3RDecoder(
    #     random_image_idx_embedding=True,
    #     enc_embed_dim=encoder.embed_dim,
    #     max_image_idx=2048,
    # )

    # ViT-B14
    # encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    # transformer = Transformer(encoder)
        #     Block(
        #         dim=embed_dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         drop=drop,
        #         attn_drop=attn_drop,
        #         norm_layer=nn.LayerNorm,
        #     attn_implementation=attn_implementation,
        # ) for _ in range(depth)

    encoder_layer = TransformerEncoderLayer(
        d_model=encoder.embed_dim,  # Match ViT-Base dimension
        nhead=12,
        dim_feedforward=3072,
        dropout=0.0,
        activation='gelu',
        batch_first=True,
    )

    transformer = TransformerEncoder(
        encoder_layer,
        # num_layers=12,
        num_layers=6,
    )
    return transformer

class Transformer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x):
        # print(x.shape)
        for blk in self.encoder.blocks:
            x = blk(x)
        return x

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
    def __init__(self, sdpa=SDPBackend.MATH, enable_timing=False):
        super().__init__()
        
        # Components
        self.encoder = get_encoder()
        self.transformer = get_global_transformer(self.encoder, sdpa)
        self.decoder = get_decoder(self.encoder)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.enable_timing = enable_timing
        self.timings = {'encoder': [], 'transformer': [], 'decoder': []} if enable_timing else None

    def _time_start(self):
        if self.enable_timing:
            torch.cuda.synchronize()
            return time.time()
        return None

    def _time_end(self, start_time, component):
        if self.enable_timing and start_time is not None:
            torch.cuda.synchronize()
            self.timings[component].append(time.time() - start_time)

    def forward(self, images):
        B, K, C, H, W = images.shape
        
        # Split frames across GPUs
        assert K % self.world_size == 0, "Number of frames must be divisible by number of GPUs"
        split_size = K // self.world_size
        images_split = torch.chunk(images, self.world_size, dim=1)
        local_images = images_split[self.rank].contiguous()

        # Reshape for ViT: (B*K_local, C, H, W)
        local_images = local_images.view(-1, C, H, W)
        
        # Time encoder
        start_time = self._time_start()
        local_tokens_multilayer = self.encoder.get_intermediate_layers(
            local_images, 
            n=self.encoder.n_blocks
        )  # Shape: (B*K_local, N, C)
        local_tokens = local_tokens_multilayer[-1]
        self._time_end(start_time, 'encoder')
        
        N = local_tokens.shape[1]
        
        # We could shard this along the batch dimensions, but for now we just recompute this on each GPU
        # Gather tokens to (B, K*N), run transformer, and scatter
        gathered_tokens = [torch.zeros_like(local_tokens) for _ in range(self.world_size)]
        dist.all_gather(gathered_tokens, local_tokens.contiguous())
        tokens = torch.cat(gathered_tokens, dim=0)  # Shape: (B*K, N, C)
        tokens = tokens.reshape(B, -1, self.encoder.embed_dim) # Shape: (B, K*N, C)

        # Time transformer
        start_time = self._time_start()
        transformed = self.transformer(tokens)
        # # For Fast3RDecoder, use the following call to add dummy positions and image ids
        # transformed = self.transformer(
            # tokens.chunk(K, dim=1), 
            # positions=torch.zeros((B, K*N, 2), device=tokens.device, dtype=tokens.dtype).chunk(K, dim=1), 
            # image_ids=torch.zeros((K, N), device=tokens.device).chunk(K, dim=0)
        # )
        self._time_end(start_time, 'transformer')
        
        transformed = transformed.view(B * K, -1, self.encoder.embed_dim) # Shape: (B*K, N, C)
        transformed_split = torch.chunk(transformed, self.world_size, dim=0)
        local_transformed = transformed_split[self.rank]
        
        # Time decoder
        start_time = self._time_start()
        tokens_multilayer = [tok for tok in local_tokens_multilayer]
        tokens_multilayer[-1] = local_transformed
        local_decoded = self.decoder(tokens_multilayer, img_info=(H, W))
        self._time_end(start_time, 'decoder')
        
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
        dtype: torch.dtype = torch.float32,
        sdpa_str: str = 'math',
        enable_timing: bool = True,
    ) -> None:
    torch.set_grad_enabled(False)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    sdpa = SDPA_TYPES[sdpa_str]

    model = ParallelFast3r(sdpa=sdpa, enable_timing=enable_timing).to(rank).eval()
    
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
        with torch.inference_mode(), \
            torch.amp.autocast('cuda', dtype=dtype), \
            torch.nn.attention.sdpa_kernel([sdpa]):
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
            print(f"Trial {trial+1}: {trial_time:.3f}s, {peak_memory:.2f}GB")
    
    if rank == 0:
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        avg_memory = sum(peak_memories) / len(peak_memories)
        
        print("\nResults:")
        print(f"Output shape: {output.shape}")
        print(f"Average forward pass time: {avg_time:.3f}s ± {std_time:.3f}s")
        
        if enable_timing:
            # Calculate component timings
            component_timings = model.module.timings
            avg_encoder_time = sum(component_timings['encoder'][num_warmup:]) / num_trials
            avg_transformer_time = sum(component_timings['transformer'][num_warmup:]) / num_trials
            avg_decoder_time = sum(component_timings['decoder'][num_warmup:]) / num_trials
            print(f"Component timing breakdown:")
            print(f"  - Encoder:     {avg_encoder_time:.3f}s ({avg_encoder_time/avg_time*100:.1f}%)")
            print(f"  - Transformer: {avg_transformer_time:.3f}s ({avg_transformer_time/avg_time*100:.1f}%)")
            print(f"  - Decoder:     {avg_decoder_time:.3f}s ({avg_decoder_time/avg_time*100:.1f}%)")
        
        print(f"Average peak GPU memory usage: {avg_memory:.2f}GB")

    dist.destroy_process_group()

# Run this on an 8 GPU machine using python main.py --batch_size 2 --image_size 224 --patch_size 16 --num_frames 16
if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    from functools import partial
    import math
    import os
    from torch.nn.functional import scaled_dot_product_attention

    # print(f"{torch.cuda.is_}") #cuda version
    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")
    print(f"{torch.cuda.get_arch_list()=}")
    print(f"{torch.cuda.is_bf16_supported()=}")

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
    parser.add_argument('--dtype', type=str, choices=['half', 'bf16', 'float'], default='half', help='Data type for processing')
    parser.add_argument('--sdpa', type=str, choices=list(SDPA_TYPES.keys()), default='math', help='SDPA type for processing')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--enable_timing', action='store_true', help='Enable component-wise timing measurements')
    args = parser.parse_args()

    batch_size = args.batch_size
    image_size = args.image_size 
    patch_size = args.patch_size
    num_frames = args.num_frames

    world_size = args.world_size
    if world_size is None:
        world_size = torch.cuda.device_count()
    print(f"Number of GPUs: {world_size}")

    if args.dtype == 'half':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(f"Running with dtype: {dtype}")

    sdpa = SDPA_TYPES[args.sdpa]
    print(f"Testing attention with sdpa: {sdpa}")
    with sdpa_kernel([sdpa]), torch.no_grad():
        x = torch.randn(1, 1, 1, 8).cuda().half()
        # B, H, N, D = 1, 12, 1024 * 1024, 768
        # x = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
        with sdpa_kernel(sdpa):
            scaled_dot_product_attention(x, x, x)
        del x
        print("... attn okay")
    
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
            dtype,
            args.sdpa,
            args.enable_timing
        ),
        nprocs=world_size,
        join=True)
