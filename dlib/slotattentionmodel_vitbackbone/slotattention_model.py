import torch.nn as nn
from dlib.slotattentionmodel_vitbackbone.vision_transformer import vit_base, vit_small, vit_tiny
from dlib.slotattentionmodel_vitbackbone.slot_attention import SlotAttentionModel
import torch
class SlotAttentionModel_ViTBackbone(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, vit_backbone, args, num_labels, **kwargs):
        super().__init__()
        #create dynamic class instance based on vit_backbone
        d = 384
        self.backbone = eval(vit_backbone)(**kwargs)
        self.model = SlotAttentionModel(
        resolution=(int(args.crop_size / self.backbone.patch_embed.patch_size), int(args.crop_size / self.backbone.patch_embed.patch_size)),
        num_slots=1000,#args.num_classes+1,
        num_iterations=3,
        empty_cache=True, hidden_dims=(d, d, d, d),
        slot_size=d,
        decoder_resolution=(14, 14),
        )
        
        self.num_labels = num_labels

    def forward(self, x):
        out = self.backbone.get_intermediate_layers(x, n=1)[0]
        out = out[:, 1:, :]  # we discard the [CLS] token
        b, h, w = x.shape[0], int(x.shape[2] / self.backbone.patch_embed.patch_size), int(x.shape[3] / self.backbone.patch_embed.patch_size)
        dim = out.shape[-1]
        out = out.reshape(b, h, w, dim)
        # out = out.reshape(b, -1, dim)
        out = out.permute(0, 3, 1, 2)
        recon_combined, recons, masks, slots, slots_intermediate = self.model(x=x, encoder_out=out)
        features = torch.ones((x.shape[0], self.num_labels)).cuda()
        
        return features, torch.ones_like(x).cuda()