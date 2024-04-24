import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = "noise_pred"
    config.z_shape = (4, 32, 32)
    config.wandb_mode = "online"

    config.joint = False  # Whether to train brain encoder jointly with the diffusion model
    config.obs_T = False  # Whether to handle x_T as observed or latent variable
    config.t_obs = 800  # 500 # < n_timestep (= 1000)
    config.obs_ratio = 0.125

    config.autoencoder = d(
        # pretrained_path="uvit/assets/stable-diffusion/autoencoder_kl_ema.pth",
        pretrained_path="uvit/assets/stable-diffusion/autoencoder_kl.pth",
        # scale_factor=0.23010,
    )

    config.train = d(
        name="default",
        n_steps=500000,
        batch_size=1024,
        mode="uncond",
        log_interval=10,
        vis_interval=1000,
        save_interval=2000,
        eval_interval=10000,
    )

    config.optimizer = d(
        name="adamw",
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(name="customized", warmup_steps=5000)

    config.brain_encoder = d(
        pretrained_path="runs/thingsmeg/small_test_F_mse-4096_ignore_subjects-True_/brain_encoder_best.pt",
        arch=d(
            seq_len=169,
            depth=2,
            D1=270,
            D2=320,
            D3=2048,
            K=32,
            n_heads=4,
            depthwise_ksize=31,
            pos_enc_type="abs",
            d_drop=0.1,
            p_drop=0.1,
        ),
    )

    config.nnet = d(
        name="uvit",
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
        use_checkpoint=True,
    )

    config.dataset = d(
        path="data/preprocessed/thingsmeg/4_autoencoder_kl",
        thingsmeg_dir="/mnt/tsukuyomi/things-meg/",
        large_test_set=False,
        chance=False,
        montage_path="nd/utils/montages/things_meg.npy",
        n_vis_samples=8,
        # cfg=True,
        # p_uncond=0.15,
    )

    config.sample = d(
        mini_batch_size=32,  # the decoder is large
        algorithm="ddpm",  # "dpm_solver",
        dpm_solver_steps=50,
        n_batches=10, # Only used for DDPM, which takes longer time to sample and thus uses RandomSampler
        n_samples=10000,  # Only used when not training brain encoder jointly
        # cfg=True,
        # scale=0.7,
        path="",
    )

    return config
