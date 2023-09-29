r"""
build a model from a configuration, for visualization purposes
"""

# ---------------------------------------------------------------------


import os
import sys
import pathlib
import argparse

# ---------------------------------------------------------------------


CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_DIR = CURRENT_DIR.parent.resolve()
sys.path.append(str(SRC_DIR))

# ---------------------------------------------------------------------

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import mvae

# ---------------------------------------------------------------------


def main(args):
    # --- argument checking
    model = args.model
    if model is None or len(model) <= 0:
        raise ValueError("you have not selected a model, available options [{0}]".format(
            list(mvae.configs_dict.keys())
        ))

    # --- check if model in configs
    model = args.model.strip().lower()

    if model not in mvae.configs_dict:
        raise ValueError(
            "could not find model [{0}], available options [{1}]".format(
                model, list(mvae.configs_dict.keys())))

    config = mvae.configs_dict[model]

    models = mvae.model_builder(config=config[mvae.constants.MODEL_STR])

    if len(args.output) <= 0:
        models.hydra.save(f"{model}.h5")
    else:
        if os.path.isdir(args.output):
            models.hydra.save(os.path.join(args.output, f"{model}.h5"))
        else:
            models.hydra.save(args.output)

    return 0

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="",
        dest="model",
        help="model to train, options: {0}".format(
            list(mvae.configs_dict.keys())))

    parser.add_argument(
        "--output",
        default="",
        dest="output",
        help="output name")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
