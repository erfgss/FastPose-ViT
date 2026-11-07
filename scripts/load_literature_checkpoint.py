"""Restore a literature-suite decoder delta with its exact baseline dependency."""
import argparse
import hashlib
from pathlib import Path
import torch
from src.models import FastPoseViT


def load_delta(path, baseline='weights/vit_b_16.pth'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    with Path(baseline).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != checkpoint['baseline_sha256']:
        raise ValueError('Baseline SHA256 differs from the experiment dependency')
    model = FastPoseViT(**checkpoint['model_config'])
    model.initialize_residual_from_baseline(baseline)
    model.decoder.load_state_dict(checkpoint['decoder'], strict=True)
    if not checkpoint['experiment_config']['geometry']:
        model.decoder.geometry_encoder.requires_grad_(False)
    return model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('delta')
    parser.add_argument('--baseline', default='weights/vit_b_16.pth')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Choose a new output filename to preserve existing weights')
    model = load_delta(args.delta, args.baseline)
    model.save(args.output)
    print(f'Saved full checkpoint: {args.output}')
