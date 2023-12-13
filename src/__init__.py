RANDOM_SEED = 42

IMAGE_SUPPORTED_EXTENSIONS = (".png", ".jpeg", ".jpg")
TENSOR_SUPPORTED_EXTENSIONS = (".npz", ".npy")

TENSORS_DICT = {
    "hific": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    "mbt2018": "analysis/layer_3/convolution:0",
    "bmshj2018":"analysis/layer_2/convolution:0",
    "b2018": "analysis/layer_2/convolution:0",
    #"ms2020": "analysis/layer_2/convolution:0",
}

MODELS_DICT = {
    "hific": {
        "variants": ["hific-lo", "hific-mi", "hific-hi"],
    },
    "bmshj2018": {
        "factorized-mse": [f"bmshj2018-factorized-mse-{i}" for i in range(1, 9)],
        "factorized-msssim": [f"bmshj2018-factorized-msssim-{i}" for i in range(1, 9)],
        "hyperprior-mse": [f"bmshj2018-hyperprior-mse-{i}" for i in range(1, 9)],
        "hyperprior-msssim": [f"bmshj2018-hyperprior-msssim-{i}" for i in range(1, 9)]
    },
    "b2018": {
        "leaky_relu-128": [f"b2018-leaky_relu-128-{i}" for i in range(1, 5)],
        "leaky_relu-192": [f"b2018-leaky_relu-192-{i}" for i in range(1, 5)],
        "gdn-128": [f"b2018-gdn-128-{i}" for i in range(1, 5)],
        "gdn-192": [f"b2018-gdn-192-{i}" for i in range(1, 5)]
    },
    "mbt2018": {
        "mean": [f"mbt2018-mean-mse-{i}" for i in range(1, 9)],
        "mean-msssim": [f"mbt2018-mean-msssim-{i}" for i in range(1, 9)]
    },
    # "ms2020": {
    #     "cc10": [f"ms2020-cc10-mse-{i}" for i in range(1, 11)],
    #     "cc8": [f"ms2020-cc8-msssim-{i}" for i in range(1, 10)]
    # },
}

MODELS_LATENTS_DICT = {
    model: TENSORS_DICT[model_class]
    for model_class, variant_dict in MODELS_DICT.items()
    for variant in variant_dict.values()
    for model in variant
    
}