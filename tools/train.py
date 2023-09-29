r"""train an image segmentation model model"""

__author__ = "Nikolas Markou"
__version__ = "1.0.0"

# ---------------------------------------------------------------------

import os
import sys
import json
import pathlib
import argparse
import subprocess

# ---------------------------------------------------------------------

CUDA_DEVICE = 0
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_DIR = CURRENT_DIR.parent.resolve() / "src"
CONFIGS_DIR = SRC_DIR / "comkardia_segmentation" / "configs"
CHECKPOINT_DIRECTORY = "/media/arxwn/external/Training"

CONFIGS = {
    os.path.basename(file_dir).split(".")[0]:
        os.path.join(str(CONFIGS_DIR), file_dir)
    for file_dir in os.listdir(str(CONFIGS_DIR))
}

os.environ["PYTHONPATH"] = str(SRC_DIR)

# ---------------------------------------------------------------------


def main(args):
    if os.path.isfile(args.model):
        with open(args.model, mode="r") as f:
            config = json.load(f)
        config_basename = os.path.basename(args.model).split(".")[0]
    else:
        # check if model in configs
        model = args.model.lower()

        if model not in CONFIGS:
            raise ValueError(
                "could not find model [{0}], available options [{1}]".format(
                    model, list(CONFIGS.keys())))

        config = CONFIGS[model]
        config_basename = os.path.basename(config).split(".")[0]

    run_name = args.run_name
    if run_name is None or len(run_name) <= 0:
        run_name = config_basename

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    """
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if args.tf_flags:
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_AUTOTUNE_THRESHOLD"] = "1"
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = "1"
        os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = "1"

    process_args = [
        sys.executable,
        "-m", "comkardia_segmentation.train",
        "--model-directory",
        os.path.join(
            args.checkpoint_directory,
            run_name),
        "--pipeline-config",
        config
    ]

    if args.weights_directory is not None and \
            len(args.weights_directory) > 0:
        process_args += [
            "--weights-directory",
            args.weights_directory
        ]

    return \
        subprocess.check_call(
            args=process_args,
            env=os.environ)

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="",
        dest="model",
        help="model to train, options: {0}".format(list(CONFIGS.keys())))

    parser.add_argument(
        "--run-name",
        default="",
        dest="run_name",
        help="how to call this specific run")

    parser.add_argument(
        "--checkpoint-directory",
        default=CHECKPOINT_DIRECTORY,
        dest="checkpoint_directory",
        help="where to save the checkpoints and intermediate results")

    parser.add_argument(
        "--weights-directory",
        default="",
        dest="weights_directory",
        help="where to load weights from")

    parser.add_argument(
        "--gpu",
        default=CUDA_DEVICE,
        dest="gpu",
        help="select gpu device")

    parser.add_argument(
        "--tf-flags",
        dest="tf_flags",
        action="store_true",
        help="enable tensorflow flags")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
