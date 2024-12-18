import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
sys.path.append("..")
# sys.path.append("../image_synthesis")
from taming.utils.misc import instantiate_from_config
from taming.models.vqgan import GumbelVQ, VQModel


from taming.models.cond_transformer import Net2NetTransformer
import os
import torchvision.transforms.functional as TF
import PIL
import taming.modules.diffusionmodules.model as vqgan_en
from modeling.codecs.base_codec import BaseCodec
from einops import rearrange
import math


class Decoder(nn.Module):
    def __init__(self, decoder, post_quant_conv, quantize, w=16, h=16):#####
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.quantize = quantize
        self.w = w
        self.h = h

    @torch.no_grad()
    def forward(self, P):
        z = self.quantize.get_codebook_entry(P, shape=(P.shape[0], self.h, self.w, -1))
        print('看看z到底啥样', z)
        quant = self.post_quant_conv(z)
        dec = self.decoder(quant)
        x = torch.clamp(dec, -1., 1.)
        x = (x + 1.)/2.
        return x

class Encoder(nn.Module):
    def __init__(self, encoder, quant_conv, quantize):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.quantize = quantize
    
    @torch.no_grad()
    def forward(self, x):
        x = 2*x - 1

        h = self.encoder(x)

        h = self.quant_conv(h)

        _, _, _, _,_,P = self.quantize(h)


        return P
class TamingScqgan(BaseCodec):
    def __init__(
            self,
            trainable=False,
            token_shape=[16, 16],
            config_path='OUTPUT/pretrained_model/taming_dvae/custom.yaml',
            ckpt_path='OUTPUT/pretrained_model/taming_dvae/last.ckpt',
            num_tokens=1024,
            quantize_number=0,
            mapping_path=None,
    ):
        super().__init__()

        # model = self.LoadModel(config_path)
        model = self.LoadModel(config_path, ckpt_path)

        self.enc = Encoder(model.encoder, model.quant_conv, model.quantize)
        self.dec = Decoder(model.decoder, model.post_quant_conv, model.quantize, token_shape[0], token_shape[1])

        self.num_tokens = num_tokens
        self.quantize_number = quantize_number
        if self.quantize_number != 0 and mapping_path != None:
            self.full_to_quantize = torch.load(mapping_path)
            self.quantize_to_full = torch.zeros(self.quantize_number) - 1
            for idx, i in enumerate(self.full_to_quantize):
                if self.quantize_to_full[i] == -1:
                    self.quantize_to_full[i] = idx
            self.quantize_to_full = self.quantize_to_full.long()

        self.trainable = trainable
        self.token_shape = token_shape
        self._set_trainable()

    # def LoadModel(self, config_path):
    def LoadModel(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)
        # model = instantiate_from_config(config.model)
        # model = Net2NetTransformer(**config.model.params)

        model = VQModel(**config.model.params)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)

        # if (isinstance(model, Net2NetTransformer)):
        # model = model.first_stage_model
        return model

    @property
    def device(self):
        # import pdb; pdb.set_trace()
        return self.enc.quant_conv.weight.device

    def preprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-255
        """
        imgs = imgs.div(255)  # map to 0 - 1
        return imgs
        # return map_pixels(imgs)

    def postprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-1
        """
        imgs = imgs * 255
        return imgs

    def get_tokens(self, imgs, **kwargs):
        imgs = self.preprocess(imgs)
        P= self.enc(imgs)

        # output = {'token': code,'P':P}
        # #添加的

        # output = {'token': rearrange(code, 'b h w -> b (h w)')}
        # print('get_tokens输出的code形状',code)
        return P
    def get_P(self,imgs,**kwargs):
        imgs=self.preprocess(imgs)
        P=self.enc(imgs)

        return P

    def decode(self,  P):

        b, n = P.shape
        P = rearrange(P, 'b (h w) -> b h w', h=int(math.sqrt(n)))

        x_rec = self.dec(P)
        x_rec = self.postprocess(x_rec)
        return x_rec
