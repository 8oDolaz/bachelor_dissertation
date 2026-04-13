"""
BC-Transformer training script with DINOv2 ViT-B/14 backbone.

Replaces ResNet18 with DINOv2 as the visual encoder backbone.
DINOv2 patch tokens are reshaped to a spatial feature map and pooled
via SpatialSoftmax, keeping the downstream pipeline identical.

Observations:
  - Low-dim: robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos
  - RGB: robot0_eye_in_hand_image (in-hand view), agentview_image (scene view)

Usage:
    python dino_transformer.py --dataset /path/to/dataset.hdf5 --output /path/to/output_dir
    python dino_transformer.py --debug
"""
import argparse
import gc
import os
import sys

# Ensure project root is importable (for models.dinov2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
import robomimic.scripts.train as _train_module

# Import DINOv2Conv so it auto-registers in robomimic's backbone registry
from models.dinov2 import DINOv2Conv  # noqa: F401


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
    Sets hyperparameters for BC-Transformer training with DINOv2 backbone.

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
    config.train.num_data_workers = 2
    config.train.hdf5_cache_mode = "low_dim"
    config.train.hdf5_use_swmr = True
    config.train.hdf5_normalize_obs = False
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.seq_length = 10                        # transformer context length

    config.train.dataset_keys = (
        "actions",
        "rewards",
        "dones",
    )

    config.train.goal_mode = None

    ## Learning ##
    config.train.cuda = True
    config.train.batch_size = 16                        # smaller batch — DINOv2 uses more memory
    config.train.num_epochs = 100
    config.train.seed = 1
    config.train.max_grad_norm = 100.0

    ### Observation Config ###

    # Low-dim: end-effector pose + gripper (NO object pose)
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]

    # RGB: in-hand camera + scene camera
    config.observation.modalities.obs.rgb = [
        "robot0_eye_in_hand_image",
        "agentview_image",
    ]

    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []

    ### RGB Encoder Config ###
    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64

    # Backbone: DINOv2 ViT-B/14
    config.observation.encoder.rgb.core_kwargs.backbone_class = "DINOv2Conv"
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = True
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze_until_block = 10

    # Pooling: SpatialSoftmax on the [768, 16, 16] patch feature map
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

    # Stochastic GMM policy
    config.algo.gmm.enabled = True
    config.algo.gmm.num_modes = 5
    config.algo.gmm.min_std = 0.0001
    config.algo.gmm.std_activation = "softplus"
    config.algo.gmm.low_noise_eval = True

    # Transformer policy
    config.algo.transformer.enabled = True
    config.algo.transformer.context_length = 10           # matches train.seq_length
    config.algo.transformer.embed_dim = 512
    config.algo.transformer.num_layers = 6
    config.algo.transformer.num_heads = 8
    config.algo.transformer.emb_dropout = 0.1
    config.algo.transformer.attn_dropout = 0.1
    config.algo.transformer.block_output_dropout = 0.1
    config.algo.transformer.sinusoidal_embedding = False
    config.algo.transformer.activation = "gelu"
    config.algo.transformer.supervise_all_steps = True
    config.algo.transformer.nn_parameter_for_timesteps = True
    config.algo.transformer.pred_future_acs = False

    return config


def get_config(dataset_path=None, output_dir=None, debug=False):
    """
    Construct config for training.

    Args:
        dataset_path (str or None): path to hdf5 dataset.
        output_dir (str): path to output folder.
        debug (bool): if True, shrink training for a quick test run.
    """
    if dataset_path is None:
        dataset_path = TestUtils.example_dataset_path()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_runs")

    config = config_factory(algo_name="bc")

    ### Experiment Config ###
    config.experiment.name = f"dino_transformer_{config.train.num_epochs}"
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

    args = parser.parse_args()

    if args.debug:
        Macros.DEBUG = True

    config = get_config(
        dataset_path=args.dataset,
        output_dir=os.path.abspath(args.output) if args.output is not None else None,
        debug=args.debug,
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    _train_module.train(config, device=device)
