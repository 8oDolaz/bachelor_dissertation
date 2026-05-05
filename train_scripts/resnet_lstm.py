"""
BC-RNN training script with image observations (no object pose).

Observations:
  - Low-dim: robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos
  - RGB: robot0_eye_in_hand_image (in-hand view), agentview_image (scene view)

Usage:
    python test_bc_rnn.py --dataset /path/to/dataset.hdf5 --output /path/to/output_dir
    python test_bc_rnn.py --debug
"""
import argparse
import gc
import os

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
from robomimic.algo import algo_factory as _original_algo_factory
from robomimic.models.base_nets import ResNet18Conv
import robomimic.scripts.train as _train_module


def freeze_backbone_except_last(model):
    """
    Freeze all ResNet18 backbone layers except the last residual block (layer4).

    ResNet18Conv.nets is a Sequential of:
        [0] conv1, [1] bn1, [2] relu, [3] maxpool,
        [4] layer1, [5] layer2, [6] layer3, [7] layer4

    We freeze [0]-[6] and leave [7] (layer4) trainable.
    The SpatialSoftmax pooling, FC projection, RNN, and policy head
    remain fully trainable as well.
    """
    frozen_count = 0
    trainable_count = 0
    for module in model.nets.modules():
        if isinstance(module, ResNet18Conv):
            children = list(module.nets.children())
            # Freeze everything except the last child (layer4)
            for child in children[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            # Count trainable params in layer4
            for param in children[-1].parameters():
                trainable_count += param.numel()
    print(f"\n[Backbone Freeze] Frozen params: {frozen_count:,} | "
          f"Trainable backbone (layer4) params: {trainable_count:,}\n")
    return model


def _algo_factory_with_freeze(*args, **kwargs):
    """Wrapper around algo_factory that freezes ResNet18 backbone after model creation."""
    model = _original_algo_factory(*args, **kwargs)
    freeze_backbone_except_last(model)
    return model


# Patch algo_factory in the train module so frozen backbone is used
_train_module.algo_factory = _algo_factory_with_freeze

# Patch save_model to force GC before each save.
# During state_dict() Python's GC may trigger and invoke robosuite EGL context
# destructors on the wrong thread, which segfaults. Collecting explicitly
# beforehand drains those pending destructors while nothing critical is happening.
_original_save_model = TrainUtils.save_model

def _save_model_gc(*args, **kwargs):
    gc.collect()
    return _original_save_model(*args, **kwargs)

TrainUtils.save_model = _save_model_gc


def set_hyperparameters(config):
    """
    Sets hyperparameters for BC-RNN training with image observations.

    Uses eef_pos, eef_quat, gripper_qpos as low-dim observations
    and in-hand + scene camera views as RGB observations (no object pose).
    """

    ## Save config ##
    config.experiment.save.enabled = True
    config.experiment.save.every_n_seconds = None
    config.experiment.save.every_n_epochs = 15
    config.experiment.save.epochs = []
    config.experiment.save.on_best_validation = False
    config.experiment.save.on_best_rollout_return = False
    config.experiment.save.on_best_rollout_success_rate = True

    # Epoch definition
    config.experiment.epoch_every_n_steps = 100
    config.experiment.validation_epoch_every_n_steps = 10

    # Environment overrides
    config.experiment.env = None
    config.experiment.additional_envs = None

    ## Rendering ##
    config.experiment.render = False
    config.experiment.render_video = True
    config.experiment.keep_all_videos = False
    config.experiment.video_skip = 5

    ## Evaluation rollouts ##
    config.experiment.rollout.enabled = True
    config.experiment.rollout.n = 10
    config.experiment.rollout.horizon = 400
    config.experiment.rollout.rate = 50
    config.experiment.rollout.warmstart = 0
    config.experiment.rollout.terminate_on_success = True

    ## Dataset loader ##
    config.train.num_data_workers = 2                   # use 2 workers for image datasets
    config.train.hdf5_cache_mode = "low_dim"            # cache only low-dim data (images too large for full cache)
    config.train.hdf5_use_swmr = True
    config.train.hdf5_normalize_obs = False
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.seq_length = 10                        # RNN sequence length

    config.train.dataset_keys = (
        "actions",
        "rewards",
        "dones",
    )

    config.train.goal_mode = None

    ## Learning ##
    config.train.cuda = True
    config.train.batch_size = 32                        # smaller batch for image training
    config.train.num_epochs = 100                       # 50 training epochs
    config.train.seed = 1
    config.train.max_grad_norm = 10.0                    # gradient clipping

    ### Observation Config ###

    # Low-dim: end-effector pose + gripper (NO object pose)
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]

    # RGB: in-hand camera + scene camera
    config.observation.modalities.obs.rgb = [
        "robot0_eye_in_hand_image",                     # in-hand / wrist view
        "agentview_image",                              # scene / third-person view
    ]

    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []

    ### RGB Encoder Config ###
    # ResNet18 backbone with SpatialSoftmax pooling for both camera views

    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64

    # Backbone: ResNet18
    config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = True
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False

    # Pooling: SpatialSoftmax
    config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0

    # Image augmentation: random crops
    config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False

    ### Algo Config ###

    # Optimization
    config.algo.optim_params.policy.learning_rate.initial = 1e-4
    config.algo.optim_params.policy.learning_rate.decay_factor = 0.1
    config.algo.optim_params.policy.learning_rate.epoch_schedule = []
    config.algo.optim_params.policy.regularization.L2 = 0.00

    # Loss weights
    config.algo.loss.l2_weight = 1.0
    config.algo.loss.l1_weight = 0.0
    config.algo.loss.cos_weight = 0.0

    # MLP layers between RNN and action output
    config.algo.actor_layer_dims = ()

    # Stochastic GMM policy
    config.algo.gmm.enabled = True
    config.algo.gmm.num_modes = 5
    config.algo.gmm.min_std = 0.0001
    config.algo.gmm.std_activation = "softplus"
    config.algo.gmm.low_noise_eval = True

    # RNN policy
    config.algo.rnn.enabled = True
    config.algo.rnn.horizon = 10                        # matches train.seq_length
    config.algo.rnn.hidden_dim = 400
    config.algo.rnn.rnn_type = "LSTM"
    config.algo.rnn.num_layers = 2
    config.algo.rnn.open_loop = False
    config.algo.rnn.kwargs.bidirectional = False

    return config


def get_config(dataset_path=None, output_dir=None, debug=False,
               exp_name=None, rollout_rate=None, rollout_n=None,
               train_filter_key=None, valid_filter_key=None):
    """
    Construct config for training.

    Args:
        dataset_path (str or None): path to hdf5 dataset.
        output_dir (str): path to output folder.
        debug (bool): if True, shrink training for a quick test run.
        exp_name (str or None): overrides experiment.name (output subfolder).
        rollout_rate (int or None): overrides experiment.rollout.rate.
        rollout_n (int or None): overrides experiment.rollout.n.
        train_filter_key (str or None): overrides train.hdf5_filter_key.
        valid_filter_key (str or None): overrides train.hdf5_validation_filter_key.
    """
    if dataset_path is None:
        dataset_path = TestUtils.example_dataset_path()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_runs")

    config = config_factory(algo_name="bc")

    ### Experiment Config ###
    config.experiment.name = f"resnet_lstm_{config.train.num_epochs}"
    config.experiment.validate = True
    config.experiment.logging.terminal_output_to_txt = False
    config.experiment.logging.log_tb = True

    ### Train Config ###
    config.train.data = dataset_path
    config.train.output_dir = output_dir

    config = set_hyperparameters(config)

    if debug:
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.save.every_n_epochs = 1
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

    # User overrides (applied last so they also win over debug defaults).
    if exp_name is not None:
        config.experiment.name = exp_name
    if rollout_rate is not None:
        config.experiment.rollout.rate = int(rollout_rate)
    if rollout_n is not None:
        config.experiment.rollout.n = int(rollout_n)
    if train_filter_key is not None:
        config.train.hdf5_filter_key = train_filter_key
    if valid_filter_key is not None:
        config.train.hdf5_validation_filter_key = valid_filter_key

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="path to input hdf5 dataset",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="path to output folder for logs, checkpoints, and videos",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="run a quick training run for debugging",
    )

    parser.add_argument("--exp_name", type=str, default=None,
                        help="override experiment.name (output subfolder)")
    parser.add_argument("--rollout_rate", type=int, default=None,
                        help="epochs between rollout evaluations")
    parser.add_argument("--rollout_n", type=int, default=None,
                        help="number of episodes per rollout evaluation")
    parser.add_argument("--train_filter_key", type=str, default=None,
                        help="HDF5 mask key for the training split")
    parser.add_argument("--valid_filter_key", type=str, default=None,
                        help="HDF5 mask key for the validation split")

    args = parser.parse_args()

    if args.debug:
        Macros.DEBUG = True

    config = get_config(
        dataset_path=args.dataset,
        output_dir=os.path.abspath(args.output) if args.output is not None else None,
        debug=args.debug,
        exp_name=args.exp_name,
        rollout_rate=args.rollout_rate,
        rollout_n=args.rollout_n,
        train_filter_key=args.train_filter_key,
        valid_filter_key=args.valid_filter_key,
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    _train_module.train(config, device=device)
