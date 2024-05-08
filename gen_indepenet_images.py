import os
import json
from pathlib import Path
import click
import numpy as np
import torch
from tqdm import tqdm
import legacy
import dnnlib
from torch_utils import gen_utils
from PIL import Image

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42)
@click.option('--centroids-path', type=str, help='Pass path to precomputed centroids to enable multimodal truncation')
@click.option('--class-samples', type=str, help='Path to JSON file containing the number of samples per class', required=True)
@click.option('--batch-gpu', help='Samples per pass, adapt to fit on GPU', type=int, default=32)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--imagenet-classes', type=str, help='Path to JSON file containing the ImageNet class index', required=True)
def generate_samples(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    centroids_path: str,
    class_samples: str,
    batch_gpu: int,
    outdir: str,
    imagenet_classes: str,
):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

    # Load class samples from JSON file
    with open(class_samples, 'r') as f:
        class_samples_dict = json.load(f)

    # Load ImageNet class index from JSON file
    with open(imagenet_classes, 'r') as f:
        imagenet_classes_dict = json.load(f)

    os.makedirs(outdir, exist_ok=True)

    for class_idx, class_info in imagenet_classes_dict.items():
        class_id, class_name = class_info
        num_samples = class_samples_dict.get(class_id, 0)
        if num_samples == 0:
            continue

        print(f'Generating {num_samples} samples for class {class_id} ({class_name})...')
        class_dir = os.path.join(outdir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        for i in tqdm(range(num_samples)):
            seed_i = seed + i
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c = torch.tensor([int(class_idx)], device=device)
            w = gen_utils.get_w_from_seed(G, 1, device, truncation_psi, seed=seed_i,
                                          centroids_path=centroids_path, class_idx=c.squeeze())
            img = gen_utils.w_to_img(G, w, to_np=True)
            img = img[0]

            image_name = f"{class_id}_{i:05d}.png"
            image_path = os.path.join(class_dir, image_name)
            Image.fromarray(img).save(image_path)

if __name__ == "__main__":
    generate_samples()

"""
python gen_indepent_images.py --outdir=samplesheet --trunc=1.0 \
--network=https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl \
--class-samples personal_storage/scout/fid-flaws/data/class_samples.json \
--imagenet-classes /personal_storage/scout/fid-flaws/imagenet_class_index.json \
--batch-gpu 32
"""