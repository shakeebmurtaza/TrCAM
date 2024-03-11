# import sys
import os
clip_seg_base_path = os.environ['CLIPES_PATH']#'/projets/AR15550/wsol/CLIP-ES/'
# sys.path.append(clip_seg_base_path)

# -*- coding:UTF-8 -*-
from dlib.clip_pytorch_grad_cam import GradCAM
import torch
# import clip
from dlib.clip_baseline import clip
from PIL import Image
import numpy as np
import cv2
from torch import nn
import pandas as pd

from tqdm import tqdm
from dlib.clip_pytorch_grad_cam.utils.image import scale_cam_image
from dlib.clip.utils import parse_xml_to_dict, scoremap2bbox
from dlib.clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY#, imagenet_templates
import argparse
from lxml import etree
import time
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

from dlib.configure import constants
from dlib.utils.tools import get_tag
from dlib.utils.shared import reformat_id
from dlib.configure.config import get_root_wsol_dataset
import yaml
from typing import Optional
# from dlib.utils.tools import get_device
from torch import device
from typing import Union, TypeVar
import json

from os.path import join

T = TypeVar('T', bound='Module')

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])

    return dataset_list

def zeroshot_classifier(classnames, templates, model, _device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(_device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(_device)
    return zeroshot_weights.t()

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs


class Get_CLIP_ATTN(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, model, args, save_cams=True):
        super(Get_CLIP_ATTN, self).__init__()
        
        self.clipmodel, _ = clip.load(os.path.join(clip_seg_base_path, 'weights/ViT-B-16.pt'), device=f'cuda:{str(args.c_cudaid)}')

        self.clipmodel.eval()
        self.clipmodel = self.clipmodel.to(f'cuda:{str(args.c_cudaid)}')
        self.save_cams = save_cams
        
        #=============get class names================
        assert args.dataset in [constants.YTOV1, constants.YTOV22, constants.OpenImages, constants.CUB, constants.ILSVRC], f"{args.dataset} dataset not supported to obtain seeds from CLIP"
        
        self.class_name = []
        if args.dataset == constants.OpenImages:
            self.classes_ids = {}
            
            root = get_root_wsol_dataset()
            root_ds = join(root, args.dataset)
            cls=pd.read_csv(join(root_ds, 'class-descriptions-boxable.csv'), header=None)
            tmp_classes_id = {i.replace('/m/', ''):j for i,j in zip(cls[0], cls[1])}
            train_cls_ids = []
            with open(join(args.metadata_root, 'train', 'image_ids.txt'), 'r') as fin:
                for line in fin.readlines():
                    class_id = line.split('/')[1]
                    if class_id in tmp_classes_id.keys() and class_id not in self.classes_ids.keys():
                        self.classes_ids[class_id] = tmp_classes_id[class_id]
                        self.class_name.append( tmp_classes_id[class_id])
        elif args.dataset == constants.CUB:
            self.classes_ids = {}
            with open(join(args.metadata_root, 'train', 'image_ids.txt'), 'r') as fin:
                for line in fin.readlines():
                    class_name = line.split('/')[0].split('.')[1].replace('_', ' ')
                    class_id = line.split('/')[0]
                    if class_id not in self.classes_ids.keys():
                        self.classes_ids[class_id] = class_name
                        self.class_name.append(class_name)
        elif args.dataset == constants.ILSVRC:
            self.classes_ids = {}
            with open(join(args.metadata_root, 'imagenet1000_clsidx_to_labels.txt'), 'r') as fin:
                imagenet_labels = list(np.array(fin.read().splitlines()))
                imagenet_labels.pop(0)
            with open(join(args.metadata_root, 'train', 'class_labels.txt'), 'r') as fin:
                for line in fin.readlines():
                    class_id = int(line.split(',')[1].replace('\n', ''))
                    class_name = imagenet_labels[class_id]
                    if class_id not in self.classes_ids.keys():
                        self.classes_ids[class_id] = class_name
                        self.class_name.append(class_name)
        else:
            root = get_root_wsol_dataset()
            root_ds = join(root, args.dataset)
            with open(join(root_ds, "class_id.yaml"), 'r') as f:
                self.classes_id = yaml.safe_load(f)
            
            for key in self.classes_id.keys():
                self.class_name.append(key)
        #=============get class names================
        
        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.clipmodel, _device=f'cuda:{str(args.c_cudaid)}')#['a rendering of a weird {}.'], model)
        # self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.clipmodel, _device=f'cuda:{str(args.c_cudaid)}')#['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(self.class_name, ['a clean origami {}.'], self.clipmodel, _device=f'cuda:{str(args.c_cudaid)}')#['a rendering of a weird {}.'], model)
        self.bg_text_features = self.bg_text_features.to(f'cuda:{str(args.c_cudaid)}')
        self.fg_text_features = self.fg_text_features.to(f'cuda:{str(args.c_cudaid)}')
        
        self.target_layers = [self.clipmodel.visual.transformer.resblocks[-1].ln_1]
        self.cam = GradCAM(model=self.clipmodel, target_layers=self.target_layers, reshape_transform=reshape_transform)

        # ## ================ For saving seed CAMS ===================
        self.save_cams_before_training = args.save_cams_before_training
        if self.save_cams_before_training:
            tag = get_tag(args, checkpoint_type=constants.BEST_LOC)
            tag += '_cams_{}'.format(constants.TRAINSET)
            self.root_dir_to_save_cam = os.path.join(constants.DATA_CAMS, tag)
            os.makedirs(self.root_dir_to_save_cam, exist_ok=True)
            print(f"Saving CAMs before training to {self.root_dir_to_save_cam} using {args.dataset} dataset and CLIP")
        
        # ## ================ For saving seed CAMS ===================
        
        
            
        self.args = args
    
    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))
    
    @staticmethod
    def _normalize(cams: torch.Tensor, spatial_dims: Optional[int] = None) -> torch.Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def forward(self, x, class_idx: int = None,
                 normalized: bool = True,
                #  reshape: Optional[Tuple] = None,
                #  argmax: Optional[bool] = False,
                img_id=None,):
        # if self.save_cams_before_training:
        
        # import time
        # start_time = time.time()
            
        self.x_in = x.clone()
        h, w = x.shape[-2], x.shape[-1]
        image_features, attn_weight_list = self.clipmodel.encode_image(x, h, w)
        
        if self.args.model['exp_label_seeting'] == 'PR':
            assert img_id is not None, "img_id should not be none"
            prompt = self.args.name_pred_cl[img_id]
        elif self.args.model['exp_label_seeting'] == 'AVG':
            prompt = self.args.model['fiexed_prompt'] 
        elif self.args.model['exp_label_seeting'] == 'GT':
        # self.clipmodel.train()
            assert img_id is not None, "img_id should not be none"
            prompt = img_id.split("/")[0] if self.args.dataset != constants.OpenImages else img_id.split("/")[1] #class_id_to_label(class_idx, self.labels)
            if self.args.dataset == constants.OpenImages:
                prompt = self.classes_ids[prompt]
            elif self.args.dataset == constants.CUB:
                class_idx_for_prompt = img_id.split('/')[0]
                prompt = self.classes_ids[class_idx_for_prompt] 
            elif self.args.dataset == constants.ILSVRC:
                prompt = self.classes_ids[class_idx]    
            assert prompt in self.class_name, f"Invalid class name for harvesting seeds from CLIP. Expected one of {self.class_name}"
        else:
            raise NotImplementedError(f"Invalid label setting {self.args.model['exp_label_seeting']}")
        prompts = [prompt]
        label_id_list = [self.class_name.index(prompt)]
        bg_features_temp = self.bg_text_features.to(f'cuda:{str(self.args.c_cudaid)}')  # [bg_id_for_each_image[im_idx]].to(device_id)
        fg_features_temp = self.fg_text_features[label_id_list].to(f'cuda:{str(self.args.c_cudaid)}')
        if 'include_prior_bg' in self.args.model['exp_seeting']:
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        else:
            # bg_features_temp = torch.ones_like(bg_features_temp)
            text_features_temp = fg_features_temp#torch.cat([fg_features_temp, bg_features_temp*0], dim=0)
            # text_features_temp = fg_features_temp
        input_tensor = [image_features, text_features_temp.to(f'cuda:{str(self.args.c_cudaid)}'), h, w]
        refined_cam_to_save = []
        for idx, label in enumerate(prompts):
            # keys.append(new_class_names.index(label))
            targets = [ClipOutputTarget(prompts.index(label))]

            #torch.cuda.empty_cache()
            with torch.set_grad_enabled(True):
                grayscale_cam, logits_per_image, attn_weight_last = self.cam(input_tensor=input_tensor,
                                                                                    targets=targets,
                                                                                    target_size=None)  # (ori_width, ori_height))

            grayscale_cam = grayscale_cam[0, :]

            grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))
            # highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))
            if 'init_cam_and_include_prior_bg_and_refine_cam' != self.args.model['exp_seeting']:
                return torch.tensor(grayscale_cam_highres).to(f'cuda:{str(self.args.c_cudaid)}')

            if idx == 0:
                attn_weight_list.append(attn_weight_last)
                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                attn_weight = torch.mean(attn_weight, dim=0)
                attn_weight = attn_weight[0].cpu().detach()
            attn_weight = attn_weight.float()

            box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
            aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
            for i_ in range(cnt):
                x0_, y0_, x1_, y1_ = box[i_]
                aff_mask[y0_:y1_, x0_:x1_] = 1

            aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
            aff_mat = attn_weight

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

            for _ in range(2):
                trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
            trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

            for _ in range(1):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            trans_mat = trans_mat * aff_mask

            cam_to_refine = torch.FloatTensor(grayscale_cam)
            cam_to_refine = cam_to_refine.view(-1,1)

            # (n,n) * (n,1)->(n,1)
            cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
            cam_refined = cam_refined.cpu().numpy().astype(np.float32)
            cam_refined_highres = scale_cam_image([cam_refined], (h, w))[0]
            refined_cam_to_save.append(torch.tensor(cam_refined_highres))

        # keys = torch.tensor(keys)
        # #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
        # highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))
        # refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))

        assert len(refined_cam_to_save) == 1, "Invalid number of prompts for harvesting seeds from CLIP. Expected 1"
        if normalized:
            out_cam = self._normalize(refined_cam_to_save[0])
            
        if self.save_cams_before_training: #and img_id.rsplit('/', 1)[0] in self.image_ids:
            assert x.shape[0] == 1, "Batch size must be 1 to save CAMs before training"
            assert type(img_id) == str, "Image ID must be a string to save CAMs before training"
            image_id_to_save = f'{reformat_id(img_id)}.pt'#'#torch.save(cam, join(fdout, f'{reformat_id(image_id)}.pt'))
            cam = out_cam.squeeze()
            cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
            cam = cam.detach().cpu()
            torch.save(cam, os.path.join(self.root_dir_to_save_cam, image_id_to_save))
            # torch.save(pred.squeeze(), os.path.join(self.root_dir_to_save_cam, img_id.replace('/', '_')+'.pt'))
        out_cam = out_cam.to(f'cuda:{str(self.args.c_cudaid)}')
        
        # total_running_tiem_in_milliseconds = (time.time() - start_time) * 1000
        # print('\n\nclass', str(self.class_name.index(prompt)) +':', total_running_tiem_in_milliseconds, f'example used: : ({img_id})')
        return out_cam
        # targets = [ClipOutputTarget(label_id_list)]
        # with torch.set_grad_enabled(True):
        #     grayscale_cam, logits_per_image, attn_weight_last = self.cam(input_tensor=input_tensor,
        #                                                                             targets=targets,
        #                                                                             target_size=None)  # (ori_width, ori_height))

        # grayscale_cam = grayscale_cam[0, :]
        # grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))
        
        # attn_weight_list.append(attn_weight_last)
        # attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
        # attn_weight = torch.stack(attn_weight, dim=0)[-8:]
        # attn_weight = torch.mean(attn_weight, dim=0)
        # attn_weight = attn_weight[0].cpu().detach()
        # attn_weight = attn_weight.float()

        # box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        # aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
        # for i_ in range(cnt):
        #     x0_, y0_, x1_, y1_ = box[i_]
        #     aff_mask[y0_:y1_, x0_:x1_] = 1

        # aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
        # aff_mat = attn_weight

        # trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        # trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

        # for _ in range(2):
        #     trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        #     trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        # trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

        # for _ in range(1):
        #     trans_mat = torch.matmul(trans_mat, trans_mat)

        # trans_mat = trans_mat * aff_mask

        # cam_to_refine = torch.FloatTensor(grayscale_cam)
        # cam_to_refine = cam_to_refine.view(-1,1)

        # # (n,n) * (n,1)->(n,1)
        # cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
        # cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        # cam_refined_highres = scale_cam_image([cam_refined], (h, w))[0]
        # out_cam = torch.tensor(cam_refined_highres)
        # if normalized:
        #     out_cam = self._normalize(out_cam)
        # return out_cam#grayscale_cam
        # highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))
        
        with torch.no_grad():
            pred = self.clipmodel(x, prompts)[0] #self.clipmodel(x.unsqueeze(0), prompts)[0]
        pred = torch.sigmoid(pred[0][0])
        if normalized:
            pred = self._normalize(pred)
        if self.save_cams_before_training: #and img_id.rsplit('/', 1)[0] in self.image_ids:
            assert x.shape[0] == 1, "Batch size must be 1 to save CAMs before training"
            assert type(img_id) == str, "Image ID must be a string to save CAMs before training"
            image_id_to_save = f'{reformat_id(img_id)}.pt'#'#torch.save(cam, join(fdout, f'{reformat_id(image_id)}.pt'))
            cam = pred.squeeze()
            cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
            cam = cam.detach().cpu()
            torch.save(cam, os.path.join(self.root_dir_to_save_cam, image_id_to_save))
            # torch.save(pred.squeeze(), os.path.join(self.root_dir_to_save_cam, img_id.replace('/', '_')+'.pt'))
        return pred
