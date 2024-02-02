import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = "noise_pred"
    config.z_shape = (4, 32, 32)  # ( c' , h' , w' )
    config.clip_img_dim = 768  # 512
    config.clip_text_dim = 768
    config.text_dim = 768  # 64  # reduce dimension
    config.data_type = 1

    config.autoencoder = d(
        pretrained_path="U-ViT/assets/stable-diffusion/autoencoder_kl.pth",
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref("text_dim"),
    )

    config.nnet = d(
        name="uvit_multi_post_ln_v1",
        img_size=32,  # 64,
        in_chans=4,
        patch_size=2,
        embed_dim=1024,  # 1536,
        depth=20,  # 30,
        num_heads=16,  # 24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        mlp_time_embed=False,
        text_dim=config.get_ref("text_dim"),
        num_text_tokens=1,  # 77,
        clip_img_dim=config.get_ref("clip_img_dim"),
        use_checkpoint=True,
    )

    config.sample = d(sample_steps=50, scale=7.0, b2i_cfg_mode="true_uncond")

    return config
