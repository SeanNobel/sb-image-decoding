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

    config.joint = False  # Whether to train brain encoder jointly with the diffusion model (must be False for Schrodinger bridge)

    config.autoencoder = d(
        # pretrained_path="uvit/assets/stable-diffusion/autoencoder_kl_ema.pth",
        pretrained_path="uvit/assets/stable-diffusion/autoencoder_kl.pth",
        # scale_factor=0.23010,
    )

    config.train = d(
        name="default",
        n_steps=500000,
        batch_size=1024,
        mode="cond",
        log_interval=10,
        vis_interval=1000,
        save_interval=2000,
        eval_interval=10000,
        accum_steps=1,
        use_ema=True,
    )

    config.optimizer = d(
        name="adamw",
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(name="customized", warmup_steps=5000)

    config.brain_encoder = d(
        config_path="configs/imageneteeg/clip.yaml",
        pretrained_path="runs/imageneteegclip/conformer_clip_temp_init-4.800_loss-clip_pos_enc-sine_abs_/brain_encoder_best.pt",
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
        z_shape=config.z_shape,
        use_checkpoint=True,
    )

    config.dataset = d(
        preproc_dir="data/preprocessed/imageneteeg/",
        preproc_name="0_init",
        n_vis_samples=8,
        p_uncond=0.15,
    )

    config.sample = d(
        mini_batch_size=32,  # the decoder is large
        algorithm="ddpm",
        # dpm_solver_steps=50,
        # n_batches=10,  # Only used for DDPM, which takes longer time to sample and thus uses RandomSampler
        # n_samples=10000,  # Only used when not training brain encoder jointly
        guidance=True,
        scale=0.7,
        path="figures/dc_samples/clip_guidance-0.7/",
    )

    return config
