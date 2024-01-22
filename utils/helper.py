import os
import subprocess
import io
import IPython.display
import numpy as np
import PIL.Image
import torch
from models import MODEL_ZOO
from models import build_generator
from utils.visualizer import fuse_images

def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    images = images.detach().cpu().numpy()
    images = (images + 1) * 255 / 2
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images


def build(model_name):
    """Builds generator and load pre-trained weights."""
    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.
    
    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')
    
    # Load pre-trained weights.
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        print("HELLO")
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')
    return generator


def synthesize(generator, num, synthesis_kwargs=None, batch_size=1, seed=0):
    """Synthesize images."""
    assert num > 0 and batch_size > 0
    synthesis_kwargs = synthesis_kwargs or dict()
    
    # Set random seed.
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Sample and synthesize.
    outputs = []
    
    for idx in range(0, num, batch_size):
        batch = min(batch_size, num - idx)
        code = torch.randn(batch, generator.z_space_dim).cuda()
        with torch.no_grad():
            images = generator(code, **synthesis_kwargs)['w']
    return images

def imshow(images, viz_size=256, col=0, spacing=0):
    """Shows images in one figure."""
    fused_image = fuse_images(
        images,
        col=col,
        image_size=viz_size,
        row_spacing=spacing,
        col_spacing=spacing
    )
    fused_image = np.asarray(fused_image, dtype=np.uint8)
    data = io.BytesIO()
    PIL.Image.fromarray(fused_image).save(data, 'jpeg')
    im_data = data.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))
    return disp

def save_img(images, viz_size=256, col=0, spacing=0):
    """Shows images in one figure."""
    fused_image = fuse_images(
        images,
        col=col,
        image_size=viz_size,
        row_spacing=spacing,
        col_spacing=spacing
    )
    fused_image = np.asarray(fused_image, dtype=np.uint8)
    img = PIL.Image.fromarray(fused_image)
    return img

def slerp(a, b, t):
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)
    d = (a * b).sum(dim=1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=1, keepdim=True)
    return d