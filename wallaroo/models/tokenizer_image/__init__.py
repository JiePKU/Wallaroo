import torch 


def build_image_tokenizer(vqvae_config):

    if vqvae_config.type == 'LlamaGen_16x16':
        from .vq_model import Llamagen_VQ_16
        vqvae_model = Llamagen_VQ_16(codebook_size=vqvae_config.params.codebook_size, codebook_embed_dim=vqvae_config.params.codebook_embed_dim)
        vqvae_checkpoint = torch.load(vqvae_config.pretrained_model_path, map_location="cpu")
        m, u = vqvae_model.load_state_dict(vqvae_checkpoint["model"])
        print(f"loading vqvae pretraining weights, missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0, f"Found unexpected keys: {u}, please check the ckpt carefully."

    elif vqvae_config.type == 'Cosmos_8x8':
        from wallaroo.models.tokenizer_image.cosmos_tokenizer.inference.image_lib import ImageTokenizer

        #  Create the ImageTokenizer instance with the encoder & decoder.
        #    - device="cuda" uses the GPU
        #    - dtype="bfloat16" expects Ampere or newer GPU (A100, RTX 30xx, etc.)
        vqvae_model = ImageTokenizer(
            checkpoint_enc=vqvae_config.encoder_ckpt,
            checkpoint_dec=vqvae_config.decoder_ckpt,
            device='cuda',
            dtype="bfloat16",
        )
    elif vqvae_config.type == "MOVQGAN_8x8":
        from .movqgan import get_movqgan_model
        vqvae_model =  get_movqgan_model(cache_dir=vqvae_config.pretrained_model_path)

    else:
        raise ValueError(f"model_type {vqvae_config.type} not supported.")
    
    return vqvae_model

