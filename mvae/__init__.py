"""Multidimensional Variation Autoencoder (mvae) package."""

__author__ = "Nikolas Markou"
__version__ = "0.1.1"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import pathlib
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .vae import VAE
from . import layer_blocks
from . import constants
from .multiscale_vae import MultiscaleVAE
from .utilities import logger, load_config

# ---------------------------------------------------------------------

current_dir = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------

configs_dir = current_dir / "configs"

configs = [
    (os.path.basename(str(c)), load_config(str(c)))
    for c in configs_dir.glob("*.json")
]

configs_dict = {
    os.path.splitext(os.path.basename(str(c)))[0]: load_config(str(c))
    for c in configs_dir.glob("*.json")
}

# ---------------------------------------------------------------------

__all__ = [
    configs,
    configs_dict,
    constants,
    VAE,
    MultiscaleVAE
]
