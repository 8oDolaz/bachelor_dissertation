"""
BC-Transformer training script with DINOv2 + VQ-VAE visual encoder.

Implements the VQ-VAE approach from Porichis et al. (Robotics 2024, 13, 98),
adapted to use DINOv2 ViT-S/14 as the image encoder (frozen except for the
last transformer block, final LayerNorm, and a trainable 1x1 projection) and
a BC-Transformer GMM policy (from robomimic) in place of the paper's LSTM
target-position decoder.

Per-camera pipeline:
  Image -> CropRandomizer
        -> DINOv2 (frozen prefix) -> last block + norm (trainable) -> 1x1 proj
        -> SpatialVectorQuantizer (EMA codebook)
        -> |- ImageDecoderDeconv (reconstruction, L_IR)
           |- Flatten + Linear(feature_dim)  --> transformer input
  ... [concat all cameras + low-dim]
        -> BC-Transformer GMM policy

Joint loss (paper Eq. 5):
    L = L_policy + beta * L_VQ + lambda * L_IR

Observations:
  - Low-dim: robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos
  - RGB: robot0_eye_in_hand_image, agentview_image

Usage:
    python dino_vqvae_transformer.py --dataset /path/to/dataset.hdf5 --output /path/to/output_dir
    python dino_vqvae_transformer.py --debug
"""
import argparse
import gc
import os
import sys

# Ensure project root is importable (for models.*)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
import robomimic.scripts.train as _train_module

# Registers DINOv2ConvLastTrainable as a backbone and VisualCoreVQVAE as an
# encoder core via robomimic's __init_subclass__ registration hooks.
from models.vqvae import (
    DINOv2ConvLastTrainable,   # noqa: F401 (import triggers registration)
    VisualCoreVQVAE,
    BC_Transformer_GMM_VQVAE,
)


# ---------------------------------------------------------------------------
# Patch algo_factory so we build the VQ-VAE variant of BC-Transformer-GMM
# ---------------------------------------------------------------------------
def _algo_factory_vqvae(algo_name, config, obs_key_shapes, ac_dim, device):
    model = BC_Transformer_GMM_VQVAE(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
    )

    vq_count = 0
    for module in model.nets.modules():
        if isinstance(module, VisualCoreVQVAE):
            vq_count += 1
            print(f"[VQ-VAE] VisualCoreVQVAE | codebook: "
                  f"{module._vq.num_embeddings} entries x {module._vq.embedding_dim} dims "
                  f"| decoder out: {module._decoder_out_size}")
    print(f"[VQ-VAE] Total VisualCoreVQVAE modules: {vq_count}\n")

    # Summary of trainable params.
    total_trainable = sum(p.numel() for p in model.nets.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.nets.parameters())
    print(f"[Params] Trainable: {total_trainable:,} / Total: {total:,}\n")

    return model


_train_module.algo_factory = _algo_factory_vqvae


# ---------------------------------------------------------------------------
# Patch save_model to GC before each save (avoids robosuite EGL segfaults).
# ---------------------------------------------------------------------------
_original_save_model = TrainUtils.save_model


def _save_model_gc(*args, **kwargs):
    gc.collect()
    return _original_save_model(*args, **kwargs)


TrainUtils.save_model = _save_model_gc


def set_hyperparameters(config):
    """
    Hyperparameters for BC-Transformer + DINOv2 + VQ-VAE training.
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

    config.experiment.env = None
    config.experiment.additional_envs = None

    ## Rendering ##
    config.experiment.render = False
    config.experiment.render_video = True
    config.experiment.keep_all_videos = False
    config.experiment.video_skip = 5

    ## Rollouts ##
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
    config.train.batch_size = 8                         # small: DINOv2 + decoder are memory-heavy
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

    # RGB: in-hand + scene
    config.observation.modalities.obs.rgb = [
        "robot0_eye_in_hand_image",
        "agentview_image",
    ]

    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []

    ### RGB Encoder Config (VQ-VAE) ###
    config.observation.encoder.rgb.core_class = "VisualCoreVQVAE"

    # Policy-facing feature dim (output of the Linear head on top of z_t).
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 128

    # Backbone: DINOv2 with last transformer block trainable + projection head.
    config.observation.encoder.rgb.core_kwargs.backbone_class = "DINOv2ConvLastTrainable"
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = True

    # VQ-VAE hyperparameters (paper: N=1024 codebook, B=10 block length).
    # We raise the block length to 64 to give the DINOv2 features more room;
    # vocabulary size N stays at 1024 as in the paper.
    config.observation.encoder.rgb.core_kwargs.vq_num_embeddings = 1024
    config.observation.encoder.rgb.core_kwargs.vq_embedding_dim = 64
    config.observation.encoder.rgb.core_kwargs.vq_commitment_cost = 0.25
    config.observation.encoder.rgb.core_kwargs.vq_decay = 0.99
    config.observation.encoder.rgb.core_kwargs.decoder_out_size = 224

    # Random crops
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
    config.algo.transformer.context_length = 10
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

    # VQ-VAE joint-loss weights (paper Eq. 5) are attached earlier in
    # get_config() because `config.algo` has its key schema locked after
    # factory creation.

    return config


def get_config(dataset_path=None, output_dir=None, debug=False,
               exp_name=None, rollout_rate=None, rollout_n=None,
               train_filter_key=None, valid_filter_key=None,
               codebook_size=None):
    if dataset_path is None:
        dataset_path = TestUtils.example_dataset_path()

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_runs",
        )

    config = config_factory(algo_name="bc")

    # Add the `vqvae` sub-config group before key-schema checks run in
    # set_hyperparameters(). `unlocked()` allows new keys to be created.
    #   beta         -> weight on commitment loss L_VQ
    #   recon_weight -> weight on reconstruction loss L_IR (lambda in paper)
    with config.unlocked():
        config.algo.vqvae.beta = 1.0
        config.algo.vqvae.recon_weight = 1.0

    config.experiment.name = f"dino_vqvae_transformer_{config.train.num_epochs}"
    config.experiment.validate = True
    config.experiment.logging.terminal_output_to_txt = False
    config.experiment.logging.log_tb = True

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
    if codebook_size is not None:
        config.observation.encoder.rgb.core_kwargs.vq_num_embeddings = int(codebook_size)

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
    parser.add_argument("--codebook_size", type=int, default=None,
                        help="VQ codebook size (vq_num_embeddings)")

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
        codebook_size=args.codebook_size,
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    _train_module.train(config, device=device)
