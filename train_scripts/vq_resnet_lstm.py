"""
VQ-BC-RNN training script with image observations.

Extends BC-RNN with a Vector Quantization bottleneck between the visual
encoder and the RNN policy. The frozen ResNet18 + SpatialSoftmax encoder
produces 64-dim features per camera, which are quantized via a learned
codebook before being fed to the LSTM.

Pipeline per camera:
  Image -> CropRandomizer -> ResNet18 (frozen except layer4)
        -> SpatialSoftmax(32 kp) -> Linear(64) -> VQ(codebook) -> ...
  ... [concat all cameras + low-dim] -> LSTM(400, 2 layers) -> GMM(5 modes)

Observations:
  - Low-dim: robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos
  - RGB: robot0_eye_in_hand_image (in-hand), agentview_image (scene)

Usage:
    python test_vq_bc_rnn.py --dataset /path/to/dataset.hdf5 --output /path/to/output_dir
    python test_vq_bc_rnn.py --debug
"""
import argparse
import sys
import os

# Add parent directory to path so we can import models module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
from robomimic.algo import algo_factory as _original_algo_factory
from robomimic.models.base_nets import ResNet18Conv
import robomimic.scripts.train as _train_module

# Import VQ components - this triggers VisualCoreVQ registration
# via EncoderCore.__init_subclass__
from models.vq import VisualCoreVQ, BC_RNN_GMM_VQ


def freeze_backbone_except_last(model):
    """
    Freeze all ResNet18 backbone layers except the last residual block (layer4).

    ResNet18Conv.nets is a Sequential of:
        [0] conv1, [1] bn1, [2] relu, [3] maxpool,
        [4] layer1, [5] layer2, [6] layer3, [7] layer4

    We freeze [0]-[6] and leave [7] (layer4) trainable.
    The SpatialSoftmax pooling, FC projection, VQ codebook, RNN, and policy
    head remain fully trainable.
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


def _algo_factory_with_vq(algo_name, config, obs_key_shapes, ac_dim, device):
    """
    Custom algo factory that creates BC_RNN_GMM_VQ directly and freezes
    the ResNet18 backbone.
    """
    model = BC_RNN_GMM_VQ(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
    )
    freeze_backbone_except_last(model)

    # Print VQ info
    vq_count = 0
    for module in model.nets.modules():
        if isinstance(module, VisualCoreVQ):
            vq_count += 1
            print(f"[VQ] Found VisualCoreVQ with codebook: "
                  f"{module.vq.num_embeddings} entries x {module.vq.embedding_dim} dims")
    print(f"[VQ] Total VisualCoreVQ modules: {vq_count}\n")

    return model


# Patch algo_factory in the train module
_train_module.algo_factory = _algo_factory_with_vq


def set_hyperparameters(config):
    """
    Sets hyperparameters for VQ-BC-RNN training with image observations.

    Same base hyperparameters as BC-RNN, with additional VQ-specific
    encoder configuration.
    """

    ## Save config ##
    config.experiment.save.enabled = True
    config.experiment.save.every_n_seconds = None
    config.experiment.save.every_n_epochs = 10
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
    config.experiment.rollout.n = 50
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
    config.train.seq_length = 10                        # RNN sequence length

    config.train.dataset_keys = (
        "actions",
        "rewards",
        "dones",
    )

    config.train.goal_mode = None

    ## Learning ##
    config.train.cuda = True
    config.train.batch_size = 32
    config.train.num_epochs = 50
    config.train.seed = 1

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
    # VisualCoreVQ: ResNet18 + SpatialSoftmax + Linear(64) + VQ

    config.observation.encoder.rgb.core_class = "VisualCoreVQ"
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

    # VQ-specific parameters
    config.observation.encoder.rgb.core_kwargs.vq_num_embeddings = 512
    config.observation.encoder.rgb.core_kwargs.vq_commitment_cost = 0.25
    config.observation.encoder.rgb.core_kwargs.vq_decay = 0.99

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
    config.algo.rnn.horizon = 10
    config.algo.rnn.hidden_dim = 400
    config.algo.rnn.rnn_type = "LSTM"
    config.algo.rnn.num_layers = 2
    config.algo.rnn.open_loop = False
    config.algo.rnn.kwargs.bidirectional = False

    return config


def get_config(dataset_path=None, output_dir=None, debug=False):
    """
    Construct config for VQ-BC-RNN training.

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
    config.experiment.name = "vq_bc_rnn_image"
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
