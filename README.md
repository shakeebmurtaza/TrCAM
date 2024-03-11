### Pytorch 1.11.0 code for:
`Leveraging Transformers for Weakly Supervised Object Localization in Unconstrained Videos`

### Issues:
Please create a github issue.

See full requirements at [./dependencies/requirements.txt](./dependencies/requirements.txt)

* Python
* [Pytorch](https://github.com/pytorch/pytorch)
* [torchvision](https://github.com/pytorch/vision)
* [Full dependencies](dependencies/requirements.txt)
* Build and install CRF:
    * Install [Swig](http://www.swig.org/index.php)
    * CRF
```shell
cdir=$(pwd)
cd dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
cd $cdir
cd dlib/crf/crfwrapper/colorbilateralfilter
swig -python -c++ colorbilateralfilter.i
python setup.py install
```

#### <a name="datasets"> Download datasets </a>:
See [folds/wsol-done-right-splits/dataset-scripts](
folds/wsol-done-right-splits/dataset-scripts). For more details, see
[wsol-done-right](https://github.com/clovaai/wsolevaluation) repo.

You can use these scripts to download the datasets: [cmds](./cmds). Use the
script [_video_ds_ytov2_2.py](./dlib/datasets/_video_ds_ytov2_2.py) to
reformat YTOv2.2.

Once you download the datasets, you need to adjust the paths in
[get_root_wsol_dataset()](dlib/configure/config.py).

#### <a name="datasets"> Run code </a>:
Download files in `download-files.txt` from google drive.

1. WSOL baselines: CAM over YouTube-Objects-v1.0 using ResNet50:
```shell
cudaid=0  # cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

getfreeport() {
freeport=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
}
export OMP_NUM_THREADS=50
export NCCL_BLOCKING_WAIT=1
plaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
getfreeport
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port=$freeport main.py --local_world_size=1 \
       --task STD_CL \
       --encoder_name resnet50 \
       --arch STDClassifier \
       --opt__name_optimizer sgd \
       --dist_backend gloo \
       --batch_size 32 \
       --max_epochs 100 \
       --checkpoint_save 100 \
       --keep_last_n_checkpoints 10 \
       --freeze_cl False \
       --freeze_encoder False \
       --support_background True \
       --method CAM \
       --spatial_pooling WGAP \
       --dataset YouTube-Objects-v1.0 \
       --box_v2_metric False \
       --cudaid $cudaid \
       --amp True \
       --plot_tr_cam_progress False \
       --opt__lr 0.001 \
       --opt__step_size 15 \
       --opt__gamma 0.9 \
       --opt__weight_decay 0.0001 \
       --exp_id cam_training
```
Continue training until the model converges. Afterward, SAVE CLIP maps of the training set for the next step. Proceed by transferring the two directories, namely 'YouTube-Objects-v1.0-resnet50-CAM-WGAP-cp_best_localization-boxv2_False' and 'YouTube-Objects-v1.0-resnet50-CAM-WGAP-cp_best_classification-boxv2_False', from the experiment folder to the 'pretrained' folder. These directories contain the optimal weights, which will be subsequently loaded by the TrCAM model.

2. TrCAM: Run:
```shell
cudaid=0  # cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

getfreeport() {
freeport=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
}
export OMP_NUM_THREADS=50
export NCCL_BLOCKING_WAIT=1
plaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
getfreeport
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port=$freeport main.py --local_world_size=1 \
        --task TrCAM \
        --encoder_name vit_small \
        --arch vit_localizer,
        --opt__name_optimizer sgd \
        --dist_backend gloo \
        --batch_size 32 \
        --max_epochs 30 \
        --checkpoint_save 100 \
        --keep_last_n_checkpoints 10" \
        --freeze_cl True \
        --support_background True \
        --method CAM \
        --seed_map_method ClipCAM \
        --spatial_pooling WGAP \
        --dataset  YouTube-Objects-v1.0 \
        --box_v2_metric False \
        --cudaid 0 \
        --amp True \
        --plot_tr_cam_progress False \
        --opt__lr 0.01 \
        --opt__step_size 15 \
        --opt__gamma 0.9 \
        --opt__weight_decay 0.0001 \
        --elb_init_t 1.0 \
        --elb_max_t 10.0 \
        --elb_mulcoef 1.01 \
        --sl_tc True \
        --sl_tc_knn 3 \
        --sl_tc_knn_mode instant \
        --sl_tc_knn_t 0.0 \
        --sl_tc_knn_epoch_switch_uniform -1 \
        --sl_tc_min_t 0.0 \
        --sl_tc_lambda 1.0 \
        --sl_tc_min 1 \
        --sl_tc_max 1 \
        --sl_tc_ksz 3 \
        --sl_tc_max_p 0.6 \
        --sl_tc_min_p 0.1 \
        --sl_tc_seed_tech seed_weighted \
        --sl_tc_use_roi True \
        --sl_tc_roi_method roi_all \
        --sl_tc_roi_min_size 0.05 \
        --crf_tc False \
        --crf_tc_lambda 2e-09 \
        --crf_tc_sigma_rgb 15.0 \
        --crf_tc_sigma_xy 100.0 \
        --crf_tc_scale 1.0 \
        --max_sizepos_tc False \
        --max_sizepos_tc_lambda 0.01 \
        --size_bg_g_fg_tc False \
        --empty_out_bb_tc False \
        --sizefg_tmp_tc False \
        --knn_tc 0 \
        --rgb_jcrf_tc False \
        --exp_id final_id \
        --calculate_metric_per_class True \
        --exploit_temporal_relationship False \
        --save_cams_before_training True \
        --use_on_the_fly_cams_after_half_epochs True \
        --search_hps True
```
