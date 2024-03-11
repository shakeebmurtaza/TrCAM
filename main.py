import comet_ml

import datetime as dt
import math
from copy import deepcopy
from os.path import join

# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from dlib.parallel import MyDDP as DDP
from dlib.process.parseit import parse_input, str2bool

from dlib.process.instantiators import get_model
from dlib.process.instantiators import get_optimizer
from dlib.utils.tools import log_device
from dlib.utils.tools import bye

from dlib.configure import constants
from dlib.learning.train_wsol import Trainer
from dlib.process.instantiators import get_loss
from dlib.process.instantiators import get_pretrainde_classifier
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.utils_checkpoints import find_last_checkpoint
from dlib.utils.utils_checkpoints import load_checkpoint_net
from dlib.utils.utils_checkpoints import load_checkpoint_optimizer
from dlib.utils.utils_checkpoints import load_checkpoint_lr_scheduler
from dlib.utils.utils_checkpoints import load_loss_t
import os
import sys

import dlib.dllogger as DLLogger
import optuna

from dlib.metrics.corloc_perclass import corloc_perlcass

def main(trial):
    
    args, args_dict = parse_input(eval=False, trial=trial)
    if input_args.hps_search_optimizer == constants.OPTIMIZER_COMET:
        trial.log_parameters(args)
    
    log_device(args)
    
    model = get_model(args)
    init_iter, checkpoint = find_last_checkpoint(
        join(args.outd_backup, args.save_dir_models), key=constants.CHP_CP)

    current_step = init_iter

    model.cuda(args.c_cudaid)
    load_checkpoint_net(network=model, s_dict=checkpoint[constants.CHP_M])

    if args.distributed:
        dist.barrier()

    model = DDP(model, device_ids=[args.c_cudaid], find_unused_parameters=True)

    best_state_dict = deepcopy(model.state_dict())

    optimizer, lr_scheduler = get_optimizer(args, model)
    load_checkpoint_optimizer(optimizer=optimizer,
                              s_dict=checkpoint[constants.CHP_O])
    load_checkpoint_lr_scheduler(lr_scheduler=lr_scheduler,
                                 s_dict=checkpoint[constants.CHP_LR])
    loss = get_loss(args)
    load_loss_t(loss, s_t=checkpoint[constants.CHP_T])

    inter_classifier = None
    if args.task in [constants.F_CL, constants.C_BOX, constants.TCAM]:
        chpts = {
            constants.F_CL: constants.BEST_LOC,
            constants.C_BOX: args.cb_pretrained_cl_ch_pt,
            constants.TCAM: args.tcam_pretrained_seeder_ch_pt
        }
        inter_classifier = get_pretrainde_classifier(
            args, pretrained_ch_pt=chpts[args.task], _model_with_backbone=model if args.model['arch'] == constants.VIT_LOCALIZER else None)
        inter_classifier.cuda(args.c_cudaid)

    trainer: Trainer = Trainer(
        args=args, model=model, optimizer=optimizer,
        lr_scheduler=lr_scheduler, loss=loss,
        classifier=inter_classifier, current_step=current_step)

    DLLogger.log(fmsg("Start init. epoch ..."))
    
    if args.save_cams_before_training:
        trainer.save_cams_before_training(constants.TRAINSET)
        return 0

    tr_loader = trainer.loaders[constants.TRAINSET]
    train_size = int(math.ceil(
        len(tr_loader.dataset) / (args.batch_size * args.num_gpus)))
    current_epoch = math.floor(current_step / float(train_size))

    trainer.evaluate(epoch=current_epoch, split=constants.VALIDSET)

    if args.is_master:
        trainer.model_selection(epoch=current_epoch, split=constants.VALIDSET)
        trainer.print_performances()
        trainer.report(epoch=0, split=constants.VALIDSET)

    DLLogger.log(fmsg("Epoch init. epoch done."))

    for epoch in range(current_epoch, trainer.args.max_epochs, 1):
        if args.distributed:
            dist.barrier()

        zepoch = epoch + 1
        DLLogger.log(fmsg(("Start epoch {} ...".format(zepoch))))
        
        if trainer.args.use_on_the_fly_cams_after_half_epochs and epoch >= trainer.args.max_epochs//2 and trainer.loaders[constants.TRAINSET].dataset.cams_paths != None:
            trainer.loaders[constants.TRAINSET].dataset.cams_paths = None
            _fmsg =  "\n=================================================================================="
            _fmsg += "\nSwitching to on the FLY pseudo-label Loader from CLIP as half of the epchs are done."
            _fmsg += "\n==================================================================================\n"
            DLLogger.log(_fmsg)

        train_performance = trainer.train(
            split=constants.TRAINSET, epoch=zepoch)

        trainer.evaluate(zepoch, split=constants.VALIDSET)

        if args.is_master:
            trainer.model_selection(epoch=zepoch, split=constants.VALIDSET)

            trainer.report_train(train_performance, zepoch,
                                 split=constants.TRAINSET)
            trainer.print_performances()
            trainer.report(zepoch, split=constants.VALIDSET)
            DLLogger.log(fmsg(("Epoch {} done.".format(zepoch))))

        trainer.adjust_learning_rate()

    if args.distributed:
        dist.barrier()

    trainer.save_best_epoch(split=constants.VALIDSET)
    trainer.capture_perf_meters()

    DLLogger.log(fmsg("Final epoch evaluation on test set ..."))

    chpts = [constants.BEST_LOC, constants.BEST_CL]
    # todo: keep only best_loc eval for tcam.

    if args.dataset == constants.ILSVRC:
        chpts = [constants.BEST_LOC]

    for eval_checkpoint_type in chpts:
        t0 = dt.datetime.now()

        DLLogger.log(fmsg('EVAL TEST SET. CHECKPOINT: {}'.format(
            eval_checkpoint_type)))

        if eval_checkpoint_type == constants.BEST_LOC:
            epoch = trainer.args.best_epoch_loc
        elif eval_checkpoint_type == constants.BEST_CL:
            epoch = trainer.args.best_epoch_cl
        else:
            raise NotImplementedError

        trainer.load_checkpoint(checkpoint_type=eval_checkpoint_type)

        trainer.evaluate(epoch,
                         split=constants.TESTSET,
                         checkpoint_type=eval_checkpoint_type,
                         fcam_argmax=False)

        if args.is_master:
            trainer.print_performances(checkpoint_type=eval_checkpoint_type)
            trainer.report(epoch, split=constants.TESTSET,
                           checkpoint_type=eval_checkpoint_type)
            trainer.save_performances(
                epoch=epoch, checkpoint_type=eval_checkpoint_type)
             
        if args.task == constants.TCAM and eval_checkpoint_type == constants.BEST_LOC:
            if args.dataset in [constants.CUB, constants.ILSVRC]:#, constants.YTOV1, constants.YTOV22]:
                best_value_for_optimizer = trainer.performance_meters[constants.TESTSET]['localization_IOU_50'].best_value
            elif args.dataset in [constants.YTOV1, constants.YTOV22]:
                corloc_perlcass_obj = corloc_perlcass(args=args, split='test')
                for label in corloc_perlcass_obj.cl_int.values():
                    corloc_avg, len_corloc_avg = corloc_perlcass_obj.evaluate(label=label, model=model)
                assert len_corloc_avg == len(corloc_perlcass_obj.cl_int.values())
                best_value_for_optimizer = corloc_avg
                DLLogger.log(f"Logging Value of CorLoc {str(corloc_avg)}")
            else:
                best_value_for_optimizer = trainer.performance_meters[constants.TESTSET]['localization'].best_value
            
                
        elif args.task == constants.STD_CL and eval_checkpoint_type == constants.BEST_CL:
            best_value_for_optimizer = trainer.performance_meters[constants.TESTSET]['classification'].best_value

        trainer.switch_perf_meter_to_captured()

        DLLogger.log("EVAL time TESTSET - CHECKPOINT {}: {}".format(
            eval_checkpoint_type, dt.datetime.now() - t0))

    if args.distributed:
        dist.barrier()
    if args.is_master:
        trainer.save_args()
        trainer.plot_perfs_meter()
        bye(trainer.args)

    # if trial is not None:
    #     DLLogger.log(fmsg(f"Returing best value of {str(best_value_for_optimizer)} to optuna..."))
    #     return best_value_for_optimizer
    if trial is not None:
        DLLogger.log(fmsg(f"Returing best value of {str(best_value_for_optimizer)} to {constants.OPTIMIZER_OPTUNA}..."))
        assert input_args.hps_search_optimizer in constants.OPTIMIZERS_FOR_HYPERPARAMS_SEARCH, "The optimizer for hyperparams search is not supported."
        
        if input_args.hps_search_optimizer == constants.OPTIMIZER_OPTUNA:
            return best_value_for_optimizer
        elif input_args.hps_search_optimizer == constants.OPTIMIZER_COMET:
            with trial.test():
                trial.log_metric('best_corloc', best_value_for_optimizer)
            # return best_value_for_optimizer

if __name__ == '__main__':
    # main()
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of the dataset.")
    parser.add_argument('--save_cams_before_training', default=False, type=str2bool,
                        help="""Whether or not to save cams before training.""")
    parser.add_argument('--search_hps', default=False, type=str2bool,
                        help="""Whether to search hps or not.""")
    parser.add_argument("--task", type=str, default=None, 
                        help="Type of the task.")
    parser.add_argument("--sl_tc_knn_mode", type=str, default=None,
                        help="Self-learning over tcam: time dependency.")
    parser.add_argument("--max_sizepos_tc", type=str2bool, default=None,
                        help="Max size pos fcams flag / tcam.")
    parser.add_argument('--only_create_optimizer_for_searching_hps', default=False, type=str2bool,
                        help="""Only create optimizer for searching hps or acutually run exp.""")
    parser.add_argument('--hps_search_optimizer', type=str, default=constants.OPTIMIZER_OPTUNA,
                        help='optimizer name for searching hps',
                        choices=[constants.OPTIMIZER_OPTUNA, constants.OPTIMIZER_COMET])
    
    input_args, _ = parser.parse_known_args()
    assert input_args.dataset in [constants.YTOV1, constants.YTOV22, constants.OpenImages], f"{input_args.dataset} dataset not supported to support TTCAM"
    dataset = input_args.dataset
    if input_args.save_cams_before_training != True and input_args.search_hps:# and input_args.task == constants.TCAM:
        assert input_args.hps_search_optimizer in constants.OPTIMIZERS_FOR_HYPERPARAMS_SEARCH, "The optimizer for hyperparams search is not supported."
        
        if input_args.hps_search_optimizer == constants.OPTIMIZER_OPTUNA:
            storage_path = f'ttcam-optuna-study_{dataset}_{input_args.task}'
            storage = optuna.storages.RDBStorage(url=f'sqlite:///{storage_path}.db', engine_kwargs={"connect_args": {"timeout": 30000}})
            study = optuna.create_study(direction='maximize', study_name=storage_path, storage=storage, load_if_exists=True)
            if input_args.only_create_optimizer_for_searching_hps:
                print('COMET study Created. Now exiting...')
                sys.exit(0)
            study.optimize(main, n_trials=1)
        
        elif input_args.hps_search_optimizer == constants.OPTIMIZER_COMET:
            project_name = f'ttcam_{dataset}_{input_args.task}'

            comet_configs_base_dir = 'config_comet'
            os.makedirs(comet_configs_base_dir, exist_ok=True)
            comet_config_file_path = os.path.join(comet_configs_base_dir, f'{project_name}_comet_config.txt')
            if os.path.exists(comet_config_file_path):
                if input_args.only_create_optimizer_for_searching_hps:
                    print('Optimizer Key Already Exists. Now exiting...')
                    sys.exit(0)
                    
                with open(comet_config_file_path) as f:
                    comet_config_id = f.readline()
                opt = comet_ml.Optimizer(comet_config_id)
                print('COMET Optimizer Status:\n', opt.status())
            else:
                paramters = {#gamma 0.1, epochs=3 #STEPLR
                    "opt__lr": {"type": "discrete", "values": [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]}, #"0.001",// 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009
                    "opt__step_size": {"type": "discrete", "values": [15, 30, 50, 70, 100]}, 
                    "opt__gamma": {"type": "discrete", "values": [0.1, 0.9]},
                    "opt__weight_decay": {"type": "discrete", "values": [0.001, 0.002, 0.003, 0.004, 0.005]}, #"0, 0.00001, 0.0001, 0.004, 0.005
                    }
                if input_args.task == constants.TCAM:
                    if input_args.sl_tc_knn_mode != constants.TIME_INSTANT:
                        paramters['sl_tc_knn'] = {"type": "discrete", "values":  [1, 2, 3, 4, 5]}
                    paramters['sl_tc_knn_t'] = {"type": "discrete", "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 15.0, 20.0]}
                    paramters['sl_tc_max_p'] = {"type": "discrete", "values": [.5, .6, .7]}
                    paramters['sl_tc_min_p'] = {"type": "discrete", "values": [.1, .2, .3, .4, .5, .6, .7, .8, .9]}
                    paramters['sl_tc_max']  = {"type": "discrete", "values": [1, 10, 100, 200, 300, 400, 500, 800, 1000]}
                    if input_args.max_sizepos_tc:
                        paramters['max_sizepos_tc_lambda'] = {"type": "discrete", "values": [0.01, 0.02, 0.03, 0.04, 0.05]}

                config = {
                        "algorithm": "bayes",
                        "spec": {
                        "objective": "maximize",
                        "retryAssignLimit": 5,
                        "metric": "test_best_corloc",},
                        "parameters": paramters,
                    }
                opt = comet_ml.Optimizer(config)
                print('COMET Optimizer Status:\n', opt.status())
                with open(comet_config_file_path, "w") as f:
                    f.write(opt.id)
                    
                if input_args.only_create_optimizer_for_searching_hps:
                    print('Optimizer Key Created. Now exiting...')
                    sys.exit(0)
                
            COMET_WORKSPACE='shakeebmurtaza'
            COMET_APIKEY='bbrVVBsFclbFud475m2L2WDYc'
            trial = opt.next(api_key=COMET_APIKEY, project_name=project_name+'-'+str(opt.id), workspace=COMET_WORKSPACE, log_code = False, auto_metric_logging=False, auto_param_logging=False, disabled=False)
            main(trial)
    else:
        main(None)