r"""train a multiscale variational autoencoder"""

import os
import sys
import argparse

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .train_loop import train_loop

# ---------------------------------------------------------------------


def main(args):
    # --- argument checking
    if not os.path.isfile(args.pipeline_config):
        raise ValueError("Pipeline configuration [{0}] is not valid".format(
            args.pipeline_config))

    if args.weights_dir is not None and len(args.weights_dir) > 0:
        if not os.path.isdir(args.weights_dir):
            raise ValueError("weights model directory must exist [{0}]".format(
                args.weights_dir))

    # --- launch train loop
    train_loop(
        pipeline_config_path=args.pipeline_config,
        model_dir=args.model_dir,
        weights_dir=args.weights_dir)

    return 0

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline-config",
        default="",
        dest="pipeline_config",
        help="Pipeline configuration path")

    parser.add_argument(
        "--model-directory",
        default="",
        dest="model_dir",
        help="Path to output model directory "
             "where event and checkpoint files will be written")

    parser.add_argument(
        "--weights-directory",
        default=None,
        dest="weights_dir",
        help="Path to existing model directory "
             "where to load weights to be fine tuned")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------