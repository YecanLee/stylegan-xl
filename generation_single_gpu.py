import os
import click
import numpy as np
import torch
import legacy
import dnnlib
from torch_utils import gen_utils
from PIL import Image
from tqdm import trange

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42)
@click.option('--centroids-path', type=str, help='Pass path to precomputed centroids to enable multimodal truncation')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num-classes', type=int, help='Number of classes to generate samples for', default=1000)
@click.option('--num-samples-per-class', type=int, help='Number of samples to generate per class', default=50)
@click.option('--batch-size', type=int, help='Batch size for generating samples', default=32)
def generate_samples(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    centroids_path: str,
    outdir: str,
    num_classes: int,
    num_samples_per_class: int,
    batch_size: int
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

    os.makedirs(outdir, exist_ok=True)

    for class_idx in trange(num_classes):
        print(f'Generating {num_samples_per_class} samples for class {class_idx})...')
        for i in range(num_samples_per_class):
            seed_i = seed + i
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c = torch.tensor([int(class_idx)], device=device)
            w = gen_utils.get_w_from_seed(G, batch_size, device, truncation_psi, seed=seed_i,
                                          centroids_path=centroids_path, class_idx=c.squeeze())
            img = gen_utils.w_to_img(G, w, to_np=True)
            img = img[0]

            image_name = f"{i:05d}.png"
            image_path = os.path.join(outdir, image_name)
            Image.fromarray(img).save(image_path)

if __name__ == "__main__":
    generate_samples()

"""
python generation_single_gpu.py --outdir=samplesheet --trunc=1.0 \
--network=https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl \
--num-classes 1000 \
--num-samples-per-class 50 \
--batch-size 32
"""
