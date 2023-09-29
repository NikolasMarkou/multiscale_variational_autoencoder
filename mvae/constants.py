# ---------------------------------------------------------------------

DEFAULT_EPSILON = 1e-4
DEFAULT_RELU_BIAS = 0.1
DEFAULT_BN_EPSILON = 1e-3
DEFAULT_LN_EPSILON = 1e-6
DEFAULT_BN_MOMENTUM = 0.99

# ---------------------------------------------------------------------

TYPE_STR = "type"
MODEL_STR = "model"
CONFIG_STR = "config"
DATASET_STR = "dataset"
PARAMETERS_STR = "parameters"

# ---------------------------------------------------------------------

HYDRA_STR = "hydra"
DECODER_STR = "decoder"
ENCODER_STR = "encoder"
DENOISER_STR = "denoiser"
BACKBONE_STR = "backbone"
NORMALIZER_STR = "normalizer"
DENORMALIZER_STR = "denormalizer"
POSTPROCESSOR_STR = "postprocessor"
VARIATIONAL_AUTOENCODER_STR = "variational_autoencoder"

# ---------------------------------------------------------------------

BATCH_SIZE_STR = "batch_size"
INPUT_SHAPE_STR = "input_shape"
INPUT_TENSOR_STR = "input_tensor"

# ---------------------------------------------------------------------


CONFIG_PATH_STR = "config.json"
SAVED_MODEL_STR = "saved_model"
MODEL_WEIGHTS_STR = "model_weights"
ONNX_MODEL_STR = "model.onnx"
TFLITE_MODEL_STR = "model.tflite"

KL_LOSS_STR = "kl_loss"
MAE_LOSS_STR = "mae_loss"
MSE_LOSS_STR = "mse_loss"
SSIM_LOSS_STR = "ssim_loss"
TOTAL_LOSS_STR = "total_loss"
REGULARIZATION_LOSS_STR = "regularization_loss"

MODEL_LOSS_FN_STR = "model"
DENOISER_LOSS_FN_STR = "denoiser"
SEGMENTATION_LOSS_FN_STR = "segmentation"

# ---------------------------------------------------------------------

SUPPORTED_IMAGE_LIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

# ---------------------------------------------------------------------
# plotting constants

DEFAULT_DPI = 100
DEFAULT_SYMMETRIC_FIGSIZE = (8, 8)
DEFAULT_NON_SYMMETRIC_FIGSIZE = (18, 6)

MODEL_HYDRA_STR = "model_hydra"
MODEL_DENOISER_STR = "model_denoiser"

MODEL_HYDRA_DEFAULT_NAME_STR = f"{MODEL_HYDRA_STR}.h5"
MODEL_DENOISER_DEFAULT_NAME_STR = f"{MODEL_DENOISER_STR}.h5"

# ---------------------------------------------------------------------

