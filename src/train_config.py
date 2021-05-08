"""Configuration of hyperparameters."""


def get_hyperparams(train_config):
    """Get hyperparams of a train config."""
    filters = ["mpl_", "ema_", "uda_", "student_", "teacher_", "finetune_"]
    is_hyperparam = lambda x: any([x.startswith(f) for f in filters])
    return {
        param: getattr(train_config, param)
        for param in dir(train_config)
        if is_hyperparam(param)
    }


# training config which has no methods
# pylint: disable=too-few-public-methods
class CIFAR10TrainConfig:
    """Train config."""

    # --------------------------------
    # model
    # --------------------------------
    image_width = 32
    image_height = 32
    image_channels = 3

    # --------------------------------
    # training
    # --------------------------------
    seed = 5
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    eval_every_epoch = 50

    # --------------------------------
    # training meta pseudo labels
    # --------------------------------
    mpl_num_augments = 2
    mpl_augment_magnitude = 16
    mpl_augment_cutout_const = 32 // 8
    mpl_augment_translate_const = 32 // 8
    mpl_augment_ops = [
        "AutoContrast",
        "Equalize",
        "Invert",
        "Rotate",
        "Posterize",
        "Solarize",
        "Color",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "Cutout",
        "SolarizeAdd",
    ]

    mpl_unlabeled_batch_size_multiplier = 8
    mpl_label_smoothing = 0.15
    uda_label_temperature = 0.7
    uda_threshold = 0.6
    uda_weight = 8.0
    uda_steps = 5000
    ema_decay = 0.995
    ema_start = 0

    student_dropout_rate = 0.2
    student_learning_rate = 0.05
    student_learning_rate_warmup = 5000
    student_learning_rate_numwait = 3000

    teacher_dropout_rate = 0.2
    teacher_learning_rate = 0.05
    teacher_learning_rate_warmup = 5000
    teacher_learning_rate_numwait = 0

    mpl_optimizer_momentum = 0.9
    mpl_optimizer_nesterov = True
    mpl_optimizer_weight_decay = 5e-4
    mpl_optimizer_grad_bound = 1e9

    # --------------------------------
    # student finetuning
    # --------------------------------
    finetune_learning_rate = 0.0001
    finetune_optimizer_momentum = 0.9
    finetune_optimizer_nesterov = True
    finetune_learning_rate_warmup = 25
    finetune_dropout_rate = 0.3


def get_train_config(name):
    """Creates a Config corresponging to the given dataset-name."""
    configs = {
        "cifar10": CIFAR10TrainConfig,
    }
    return configs[name]()
