from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import subprocess
from pathlib import Path
import datetime as dt
import argparse
import more_itertools as mit

import numpy as np
from tqdm import tqdm
import pretrainedmodels.utils
import yaml
import munch
import pickle as pkl
from texttable import Texttable

import torch
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt

# root_dir = dirname(dirname(dirname(abspath(__file__))))
root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)

from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from  dlib.configure import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_tag
from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import log_device
from dlib.configure import config
from dlib.utils.utils_checkpoints import find_last_checkpoint

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model

from dlib.datasets.wsol_loader import get_data_loader
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets.wsol_loader import get_class_labels
from dlib.datasets.wsol_loader import get_image_ids
from dlib.learning.train_wsol import Basic, PerformanceMeter
from dlib.crf import DenseCRFFilter

core_pattern = 'passed.txt'

virenv = "\nCONDA_BASE=$(conda info --base) \n" \
         "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
         "conda activate {}\n".format(constants._ENV_NAME)


PREAMBULE = "#!/usr/bin/env bash \n {}".format(virenv)
PREAMBULE += '\n# ' + '=' * 78 + '\n'
PREAMBULE += 'cudaid=$1\nexport CUDA_VISIBLE_DEVICES=$cudaid\n\n'


def serialize_perf_meter(performance_meters, subtrainer, split) -> dict:
    out = dict()
    for _split in subtrainer._SPLITS:
        out[_split] = {
            metric: performance_meters[_split][metric] if
            isinstance(performance_meters[_split][metric], dict) else
            vars(performance_meters[_split][metric])
            for metric in subtrainer._EVAL_METRICS
        }

    return out


def save_performances(args, performance_meters, subtrainer, split, epoch=None,
                      checkpoint_type=None):

    tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)
    tagargmax = ''
    log_path = join(args.outd, 'performance_log{}{}.pickle'.format(
        tag, tagargmax))

    with open(log_path, 'wb') as f:
        pkl.dump(serialize_perf_meter(performance_meters, subtrainer,
                                      split), f)

    log_path = join(args.outd, 'performance_log{}{}.txt'.format(
        tag, tagargmax))
    with open(log_path, 'w') as f:
        f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
            checkpoint_type, epoch, tagargmax))

        for _split in subtrainer._SPLITS:
            for metric in subtrainer._EVAL_METRICS:

                if isinstance(performance_meters[_split][
                                          metric], dict):
                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, _split, metric,
                                      performance_meters[_split][
                                          metric]['current_value']))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, _split, metric,
                                    performance_meters[
                                        _split][metric]['best_value']))
                else:
                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, _split, metric,
                                      performance_meters[_split][
                                          metric].current_value))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, _split, metric,
                                    performance_meters[
                                        _split][metric].best_value))


def cl_forward(args, model, images: torch.Tensor):
    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task in [constants.F_CL, constants.TCAM]:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits


def _compute_accuracy(args, model, loader_split):
    assert not args.distributed

    num_correct = 0
    num_images = 0

    id_pred_cl = dict()
    n = len(loader_split)

    for i, (images, targets, image_ids, _, _, _, _, _) in tqdm(enumerate(
            loader_split), ncols=80, total=n):
        images = images.cuda(args.c_cudaid)
        targets = targets.cuda(args.c_cudaid)
        with torch.no_grad():
            with autocast(enabled=args.amp_eval):
                cl_logits = cl_forward(args=args, model=model, images=images,
                ).detach()

            pred = cl_logits.argmax(dim=1)
            num_correct += (pred == targets).sum().detach()
            num_images += images.size(0)

            for id_, pl in zip(image_ids, pred):
                id_pred_cl[id_] = pl.item()

    classification_acc = num_correct / float(num_images) * 100

    torch.cuda.empty_cache()
    return classification_acc.item(), id_pred_cl


def get_name_cl(class_id: dict, label: int):
    for k in class_id:
        if class_id[k] == label:
            return k

    raise ValueError(f'label name {label} not found.')

def fast_eval():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--exp_path", type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')

    parsedargs = parser.parse_args()
    split = parsedargs.split
    label = parsedargs.label
    exp_path = parsedargs.exp_path
    tmp_outd = parsedargs.tmp_outd
    checkpoint_type = parsedargs.checkpoint_type
    assert os.path.isdir(exp_path)
    assert split == constants.TESTSET, split

    assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

    _CODE_FUNCTION = 'fast_eval_{}'.format(split)

    _VERBOSE = True
    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

        # todo: update this if needed.
        args.sl_tc_knn_t = 0.0
        args.sl_tc_min_t = 0.0
        args.sl_tc_knn = 0
        args.sl_tc_knn_mode = constants.TIME_INSTANT
        args.sl_tc_knn_epoch_switch_uniform = -1
        args.sl_tc_seed_tech = constants.SEED_UNIFORM

    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    if checkpoint_type == constants.BEST_LOC:
        epoch = args.best_epoch_loc
    elif checkpoint_type == constants.BEST_CL:
        epoch = args.best_epoch_cl
    else:
        raise NotImplementedError

    # cams.
    tag = get_tag(args, checkpoint_type=checkpoint_type)
    tag_cam = tag + '_cams_{}'.format(split)
    if split == constants.TRAINSET:

        cams_fd = join(root_dir, constants.DATA_CAMS, tag_cam)
        os.makedirs(cams_fd, exist_ok=True)
        cams_roi_file = join(root_dir, constants.DATA_CAMS, tag_cam + '.txt')

    msg = 'Task: {} \t box_v2_metric: {} \t' \
          'Dataset: {} \t Method: {} \t ' \
          'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
                                  args.method, args.model['encoder_name'])

    log_backends = [
        # ArbJSONStreamBackend(Verbosity.VERBOSE,
        #                      join(outd, "log.json")),
        # ArbTextStreamBackend(Verbosity.VERBOSE,
        #                      join(outd, "log.txt")),
    ]

    if _VERBOSE:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
    DLLogger.init_arb(backends=log_backends, is_master=True, reset=False)
    DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))
    DLLogger.log(fmsg("Evaluate epoch {}, split {}".format(epoch, split)))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_device(parsedargs)
    model = get_model(args, eval=True)
    tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
    path = join(exp_path, tag)
    _, ckpt = find_last_checkpoint(path, constants.CHP_BEST_M)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.decoder.load_state_dict(ckpt['decoder'])
    model.segmentation_head.load_state_dict(ckpt['segmentation_head'])
    model.classification_head.load_state_dict(ckpt['classification_head'])
    model.cuda()
    model.eval()

    basic_config = config.get_config(ds=args.dataset)

    if split == constants.TESTSET_VIDEO_DEMO:
        basic_config['data_paths'][split] = basic_config['data_paths'][
            constants.TESTSET]
        args.std_cams_thresh_file[split] = ''


    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']

    folds_path = join(root_dir, args.metadata_root)
    path_class_id = join(folds_path, 'class_id.yaml')
    cl_int = None
    if os.path.isfile(path_class_id):
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

    cl_name = get_name_cl(class_id=cl_int, label=int(label))
    outd = join(args.outd, checkpoint_type, split,
                f'splitted_perf_cl_{cl_name}')
    args.outd = outd

    os.makedirs(outd, exist_ok=True)

    metadata = configure_metadata(join(args.metadata_root, split))
    image_labels: dict = get_class_labels(metadata)

    image_ids = []
    for k in image_labels:
        if image_labels[k] == int(label):
            image_ids.append(k)

    loaders, _ = get_data_loader(
        args=args,
        data_roots=args.data_paths,
        metadata_root=args.metadata_root,
        batch_size=32,
        workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        proxy_training_set=False,
        dataset=args.dataset,
        num_val_sample_per_class=0,
        std_cams_folder=None,
        get_splits_eval=[split],
        isdistributed=False,
        image_ids=image_ids
    )

    subtrainer: Basic = Basic(args=args)
    basic_perf_mtr = subtrainer._set_performance_meters()

    print(f'{split} loader length: {len(loaders[split].dataset)} for class {label}: {cl_name}')
    cam_computer = CAMComputer(
        args=deepcopy(args),
        model=model,
        loader=loaders[split],
        metadata_root=os.path.join(args.metadata_root, split),
        mask_root=args.mask_root,
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset,
        split=split,
        cam_curve_interval=args.cam_curve_interval,
        multi_contour_eval=args.multi_contour_eval,
        out_folder=outd
    )

    # load stored performance meters.
    _tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)
    log_path = join(exp_path, 'performance_log{}{}.pickle'.format(
        _tag, ''))
    with open(log_path, 'rb') as fpmtr:
        performance_meters = pkl.load(fpmtr)
        # erase old.
        performance_meters[split]: dict = {
            metric: PerformanceMeter(split, higher_is_better=False
            if metric == 'loss' else True) for metric in
            subtrainer._EVAL_METRICS
        }

    print(f'EVAL {split}: classification performance')

    accuracy, id_pred_cl = _compute_accuracy(args, model,
                                             loader_split=loaders[split])
    performance_meters[split][constants.CLASSIFICATION_MTR].update(
        accuracy)

    with open(join(outd, 'class_pred.yaml'), 'w') as fp:
        yaml.dump(id_pred_cl, fp)

    print(f'EVAL {split}: cam localization performance')

    assert checkpoint_type == constants.BEST_LOC
    path_thres = join(exp_path, checkpoint_type, constants.TESTSET,
                      f'thresholds-{checkpoint_type}.yaml')
    assert os.path.isfile(path_thres)

    with open(path_thres, 'r') as fthresh:
        stuff = yaml.safe_load(fthresh)
        _iou_threshold_list = stuff['iou_threshold_list']
        _best_tau_list = stuff['best_tau_list']

    cam_performance = cam_computer.compute_and_evaluate_cams()

    nbr = len(loaders[split].dataset)

    # pred_classes.
    pred_cl = None
    if checkpoint_type == constants.BEST_LOC:
        # use classes produced by BEST_CL.
        pass

    if args.multi_iou_eval or (args.dataset == constants.OpenImages):
        loc_score = np.average(cam_performance)
    else:
        loc_score = cam_performance[args.iou_threshold_list.index(50)]

    performance_meters[split][constants.LOCALIZATION_MTR].update(loc_score)

    # dump perf in root exp.
    per_class_perf = join(exp_path, 'corloc.yaml')
    if os.path.isfile(per_class_perf):
        with open(per_class_perf, 'r') as corloc:
            stats = yaml.safe_load(corloc)

            stats['corloc'][cl_name] = loc_score.item()

            stats['corloc_avg'] = [stats['corloc'][k] for k in stats['corloc']]
            stats['corloc_avg'] = sum(stats['corloc_avg']) / float(
                len(stats['corloc_avg']))

        with open(per_class_perf, 'w') as corloc:
            yaml.dump(stats, corloc)

    else:
        with open(per_class_perf, 'w') as corloc:
            stats = {
                'dataset': args.dataset,
                'split': split,
                'method': args.method,
                'task': args.task,
                'corloc_avg': loc_score.item(),
                'corloc': {
                    cl_name: loc_score.item()
                }
            }
            yaml.dump(stats, corloc)

    dataset = args.dataset

    if dataset in [constants.CUB, constants.ILSVRC, constants.YTOV1,
                   constants.YTOV22]:
        for idx, IOU_THRESHOLD in enumerate(args.iou_threshold_list):
            performance_meters[split][
                f'{constants.LOCALIZATION_MTR}_IOU_{IOU_THRESHOLD}'].update(
                cam_performance[idx])

            performance_meters[split][
                'top1_loc_{}'.format(IOU_THRESHOLD)].update(
                cam_computer.evaluator.top1[idx])

            performance_meters[split][
                'top5_loc_{}'.format(IOU_THRESHOLD)].update(
                cam_computer.evaluator.top5[idx])

    curve_top_1_5 = cam_computer.evaluator.curve_top_1_5

    with open(join(outd, 'curves_top_1_5.pkl'), 'wb') as fc:
        pkl.dump(curve_top_1_5, fc, protocol=pkl.HIGHEST_PROTOCOL)

    title = get_tag(args, checkpoint_type=checkpoint_type)
    title = 'Top1/5: {}'.format(title)

    title += '_argmax_false'
    plot_perf_curves_top_1_5(curves=curve_top_1_5, fdout=outd,
                             title=title)

    save_performances(args, performance_meters, subtrainer, split=split,
                      epoch=epoch, checkpoint_type=checkpoint_type)

    with open(join(outd, f'thresholds-{checkpoint_type}.yaml'),
              'w') as fth:
        yaml.dump({
            'iou_threshold_list':
                cam_computer.evaluator.iou_threshold_list,
            'best_tau_list': cam_computer.evaluator.best_tau_list
        }, fth)

    DLLogger.log(fmsg('Time: {}'.format(dt.datetime.now() - t0)))
    
    draw_samples = False
    if draw_samples:
        cam_computer.out_folder = join('eval_per_class_best_exp_samples', f'best_pred_cls_{str(label)}')
        cam_computer.draw_some_best_pred(rename_ordered=True, store_imgs=True, compress=False)
        os.makedirs(cam_computer.out_folder, exist_ok=True)


def plot_perf_curves_top_1_5(curves: dict, fdout: str, title: str):

    x_label = r'$\tau$'
    y_label = 'BoxAcc'

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

    for i, top in enumerate(['top1', 'top5']):

        iouthres = sorted(list(curves[top].keys()))
        for iout in iouthres:
            axes[0, i].plot(curves['x'], curves[top][iout],
                            label=r'{}: $\sigma$={}'.format(top, iout))

        axes[0, i].xaxis.set_tick_params(labelsize=5)
        axes[0, i].yaxis.set_tick_params(labelsize=5)
        axes[0, i].set_xlabel(x_label, fontsize=8)
        axes[0, i].set_ylabel(y_label, fontsize=8)
        axes[0, i].grid(True)
        axes[0, i].legend(loc='best')
        axes[0, i].set_title(top)

    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                dpi=300)


if __name__ == '__main__':
    # split = constants.TESTSET
    # method = constants.METHOD_LAYERCAM
    # task = constants.TCAM
    # dataset = constants.YTOV22
    
    split = constants.TESTSET
    method = constants.VIT_LOCALIZER
    task = constants.TCAM
    dataset = constants.YTOV1

    fast_eval()


