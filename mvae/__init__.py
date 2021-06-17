"""Multidimensional Variation Autoencoder (mvae) package."""

__author__ = "Nikolas Markou"
__version__ = "0.1.1"
__license__ = "MIT"

from .vae import VAE
from . import layer_blocks
from .multiscale_vae import MultiscaleVAE

__all__ = [
    VAE,
    layer_blocks,
    MultiscaleVAE
]
