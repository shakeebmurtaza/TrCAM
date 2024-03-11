# import sys
# clip_seg_base_path = '/projets/AR15550/wsol/clipseg'
# sys.path.append(clip_seg_base_path)

# from models.clipseg import CLIPDensePredT
# import os
# import torch
# import torch.nn as nn
# from typing import Optional
# # from dlib.utils.tools import get_device
# from torch import device
# from typing import Union, TypeVar
# from dlib.configure.config import get_root_wsol_dataset
# import yaml
# from os.path import join
# from dlib.configure import constants
# from dlib.utils.tools import get_tag
# from dlib.utils.shared import reformat_id

# T = TypeVar('T', bound='Module')

# class Get_CLIP_ATTN(nn.Module):
#     """Linear layer to train on top of frozen features"""
#     def __init__(self, model, args, save_cams=True):
#         super(Get_CLIP_ATTN, self).__init__()

#         self.clipmodel = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
#         self.clipmodel.eval()
#         self.clipmodel.load_state_dict(torch.load(os.path.join(clip_seg_base_path, 'weights/rd64-uni.pth'), map_location=torch.device('cpu')), strict=False)
#         self.clipmodel.to(f'cuda:{str(args.c_cudaid)}')
#         self.save_cams = save_cams
#         # ## ================ For saving seed CAMS ===================
#         self.save_cams_before_training = args.save_cams_before_training
#         if self.save_cams_before_training:
#             tag = get_tag(args, checkpoint_type=constants.BEST_LOC)
#             tag += '_cams_{}'.format(constants.TRAINSET)
#             self.root_dir_to_save_cam = os.path.join(constants.DATA_CAMS, tag)
#             os.makedirs(self.root_dir_to_save_cam, exist_ok=True)
#         # train_img_id_path = os.path.join(args.metadata_root, constants.TRAINSET, 'image_ids.txt')
#         # self.image_ids = []
#         # with open(train_img_id_path) as f:
#         #     for line in f.readlines():
#         #         self.image_ids.append(line.strip('\n'))
#         # ## ================ For saving seed CAMS ===================
        
#         assert args.dataset in [constants.YTOV1, constants.YTOV22], f"{args.dataset} dataset not supported to obtain seeds from CLIP"
#         root = get_root_wsol_dataset()
#         root_ds = join(root, args.dataset)
#         with open(join(root_ds, "class_id.yaml"), 'r') as f:
#             classes_id = yaml.safe_load(f)
        
#         self.class_name = []
#         for key in classes_id.keys():
#             self.class_name.append(key)
    
#     def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
#         r"""Moves all model parameters and buffers to the GPU.

#         This also makes associated parameters and buffers different objects. So
#         it should be called before constructing optimizer if the module will
#         live on GPU while being optimized.

#         .. note::
#             This method modifies the module in-place.

#         Args:
#             device (int, optional): if specified, all parameters will be
#                 copied to that device

#         Returns:
#             Module: self
#         """
#         return self._apply(lambda t: t.cuda(device))
    
#     @staticmethod
#     def _normalize(cams: torch.Tensor, spatial_dims: Optional[int] = None) -> torch.Tensor:
#         """CAM normalization"""
#         spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
#         cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
#         cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

#         return cams

#     def forward(self, x, class_idx: int = None,
#                  normalized: bool = True,
#                 #  reshape: Optional[Tuple] = None,
#                 #  argmax: Optional[bool] = False,
#                 img_id=None,):
#         # if self.save_cams_before_training:
            
#         self.x_in = x.clone()
#         self.clipmodel.eval()
#         prompt = img_id.split("/")[0] #class_id_to_label(class_idx, self.labels)
#         assert prompt in self.class_name, f"Invalid class name for harvesting seeds from CLIP. Expected one of {self.class_name}"
#         prompts = [prompt]
#         with torch.no_grad():
#             pred = self.clipmodel(x, prompts)[0] #self.clipmodel(x.unsqueeze(0), prompts)[0]
#         pred = torch.sigmoid(pred[0][0])
#         if normalized:
#             pred = self._normalize(pred)
#         if self.save_cams_before_training: #and img_id.rsplit('/', 1)[0] in self.image_ids:
#             assert x.shape[0] == 1, "Batch size must be 1 to save CAMs before training"
#             assert type(img_id) == str, "Image ID must be a string to save CAMs before training"
#             image_id_to_save = f'{reformat_id(img_id)}.pt'#'#torch.save(cam, join(fdout, f'{reformat_id(image_id)}.pt'))
#             cam = pred.squeeze()
#             cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
#             cam = cam.detach().cpu()
#             torch.save(cam, os.path.join(self.root_dir_to_save_cam, image_id_to_save))
#             # torch.save(pred.squeeze(), os.path.join(self.root_dir_to_save_cam, img_id.replace('/', '_')+'.pt'))
#         return pred
