import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..modules.movq_modules import MOVQDecoder
from einops import rearrange
from ..modules.vqvae_blocks import Encoder
from ..modules.quantize import VectorQuantizer2


class MOVQ(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def decode_code(self, code_b):
        quant = self.quantize.embedding(code_b.flatten())
        grid_size = int((quant.shape[0])**0.5)
        quant = quant.view((1, grid_size, grid_size, 4))
        quant = rearrange(quant, 'b h w c -> b c h w').contiguous()
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def forward(self, input):
        _, diff, [_, _, index] = self.encode(input)
        dec = self.decode_code(index)
        return dec, diff
    


