

import os
from functools import partial
from dlib.configure import constants
import torch
import torch.nn as nn
from dlib.utils.tools import Dict2Obj
from typing import Optional
from dlib.unet.decoder import DecoderBlock
# from dlib.utils.tools import get_device
from dlib.visiontransformer.vision_transformer import vit_small, vit_base, vit_tiny
from typing import Union, TypeVar
T = TypeVar('T', bound='Module')

vits = {'vit_small': vit_small, 'vit_base': vit_base, 'vit_tiny': vit_tiny}

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class LocalizationHead(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LocalizationHead, self).__init__()
        self.num_labels = num_labels
        self.head = nn.Conv2d(dim, num_labels, kernel_size=3, stride=1, padding=1)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()
        
        self.decoder1 = DecoderBlock(dim, 0, 192)
        self.decoder2 = DecoderBlock(192, 0, 96)
        self.decoder3 = DecoderBlock(96, 0, 48)
        self.decoder4 = DecoderBlock(48, 0, 2)

    def forward(self, x):
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        return x
    
class ViT_Classifer(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, args, num_labels=1000):#, freeze_cl=False):
        super(ViT_Classifer, self).__init__()
        self.p = Dict2Obj(args.model)

        # arch = self.p.arch if args.task == 'STD_CL' else self.p.encoder_name
        self.encoder =  vits[self.p.encoder_name](self.p.ssl_patch_size)
        self.encoder.eval()
        embed_dim = self.encoder.embed_dim * (self.p.ssl_cl_n_last_blocks + int(self.p.ssl_cl_avgpool_patchtokens))
        
        self.classification_head = LinearClassifier(embed_dim, num_labels)
        self.support_background = True
        
    def set_classifier_frozen_flag(self, freeze_cl: bool = True):
        self.freeze_cl = freeze_cl
    
    def freeze_classifier(self):
        assert self.freeze_cl

        for module in (self.encoder.modules()):

            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

        for module in (self.classification_head.modules()):
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def assert_cl_is_frozen(self):
        assert self.freeze_cl

        for module in (self.encoder.modules()):
            for param in module.parameters():
                assert not param.requires_grad

            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

            if isinstance(module, torch.nn.Dropout):
                assert not module.training

        for module in (self.classification_head.modules()):
            for param in module.parameters():
                assert not param.requires_grad

            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

            if isinstance(module, torch.nn.Dropout):
                assert not module.training

    def forward(self, x):

        avgpool, n_last_blocks = (True, 1) if self.p.arch == constants.VIT_BASE else (False, 4)
        
        self.encoder.eval()
        with torch.no_grad():
            intermediate_output = self.encoder.get_intermediate_layers(x, n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)

        return self.classification_head(output)
    

class ViT_Localizer(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, args, num_labels):
        super(ViT_Localizer, self).__init__()

        self.p = Dict2Obj(args.model)

        self.classification_head = ViT_Classifer(args, args.num_classes)
        
        self.encoder = self.classification_head.encoder#ViT_Classifer(args, args.num_classes)
        
        self.segmentation_head = LocalizationHead(1182, num_labels)
        self.decoder = self.segmentation_head#LocalizationHead(1182, num_labels)
        self.reconstruction_head = None

    def forward(self, x, normalized=False):
        self.x_in = x.clone()
        
        self.classification_head.eval()
        self.classification_head.encoder.eval()
        with torch.no_grad():
            attn = self.classification_head.encoder.get_last_selfattention(x)
            logits = self.classification_head(x)

        w_featmap = x.shape[-2] // self.classification_head.encoder.patch_embed.patch_size
        h_featmap = x.shape[-1] // self.classification_head.encoder.patch_embed.patch_size
        nh = attn.shape[1]

        maps_dim = attn.shape[2]
        
        attn = attn[:, :, :, 1:].reshape(-1, nh*(maps_dim), w_featmap, h_featmap)

        reconst_image = None
        attn = self.segmentation_head(attn)
        
        self.cams = attn
        return logits, attn, reconst_image
        
class ViT_Get_Attn(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, model, args):
        super(ViT_Get_Attn, self).__init__()

        self.classifier = model

        self.model = model.encoder

    def forward(self, x, normalized: bool = True):
            
        self.model.eval()
        with torch.no_grad():
            attn = self.model.get_last_selfattention(x)

        #compute shapes
        w_featmap = x.shape[-2] // self.model.patch_embed.patch_size
        h_featmap = x.shape[-1] // self.model.patch_embed.patch_size
        nh = attn.shape[1]

        #Get attention map from [cls] token from the last attention layer
        attn = attn[:, :, 0, 1:].reshape(-1, nh, w_featmap, h_featmap) 
        attn = attn.sum(dim=1, keepdim=True)
        if normalized:
            attn = self.model._normalize(attn)

        return attn
