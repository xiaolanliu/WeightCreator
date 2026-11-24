"""Portable loader for the frozen local Qwen3.5-0.8B backbone."""
import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

def load(size='0.8B'):
    if size != '0.8B':
        raise ValueError('WeightCreator v1 uses only Qwen3.5-0.8B')
    path=os.environ.get('WEIGHTCREATOR_MODEL_DIR')
    if not path:
        raise RuntimeError('Set WEIGHTCREATOR_MODEL_DIR to a local Qwen3.5-0.8B checkpoint')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the backbone')
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    return model,processor
