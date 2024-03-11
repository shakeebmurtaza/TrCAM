from os.path import join, dirname, abspath, basename
from dlib.datasets.wsol_loader import get_class_labels
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets.wsol_loader import get_data_loader
import os
import yaml
from copy import deepcopy
from  dlib.configure import constants
import numpy as np

from dlib.learning.inference_wsol import CAMComputer

class corloc_perlcass:
    def __init__(self, args, split = 'test') -> None:
        self.args = args
        self.split = split
        metadata = configure_metadata(join(args.metadata_root, split))
        self.image_labels: dict = get_class_labels(metadata)
        
        # folds_path = join(self.args.root_dir, args.metadata_root)
        path_class_id = join(args.metadata_root, 'class_id.yaml')
    
        self.cl_int = None
        if os.path.isfile(path_class_id):
            with open(path_class_id, 'r') as fcl:
                self.cl_int = yaml.safe_load(fcl)
        
    def get_name_cl(self, class_id: dict, label: int):
        for k in class_id:
            if class_id[k] == label:
                return k

        raise ValueError(f'label name {label} not found.')
    
    def evaluate(self, label, model) -> None:
        image_ids = []
        for k in self.image_labels:
            if self.image_labels[k] == int(label):
                image_ids.append(k)
                
        loaders, _ = get_data_loader(
        args=self.args,
        data_roots=self.args.data_paths,
        metadata_root=self.args.metadata_root,
        batch_size=32,
        workers=self.args.num_workers,
        resize_size=self.args.resize_size,
        crop_size=self.args.crop_size,
        proxy_training_set=False,
        dataset=self.args.dataset,
        num_val_sample_per_class=0,
        std_cams_folder=None,
        get_splits_eval=[self.split],
        isdistributed=False,
        image_ids=image_ids
        )
        
        cl_name = self.get_name_cl(class_id=self.cl_int, label=int(label))
        
        print(f'Cacluating CorLoc with {self.split} loader length: {len(loaders[self.split].dataset)} for class {label}: {cl_name}')
        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=model,
            loader=loaders[self.split],
            metadata_root=os.path.join(self.args.metadata_root, self.split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=self.split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=self.args.outd
        )
        
        cam_performance = cam_computer.compute_and_evaluate_cams()
        
        if self.args.multi_iou_eval or (self.args.dataset == constants.OpenImages):
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]
        
            # dump perf in root exp.
        per_class_perf = join(self.args.outd, 'corloc_log.yaml')
        if os.path.isfile(per_class_perf):
            with open(per_class_perf, 'r') as corloc:
                stats = yaml.safe_load(corloc)

                stats['corloc'][cl_name] = loc_score.item()

                stats['corloc_avg'] = [stats['corloc'][k] for k in stats['corloc']]
                stats['corloc_avg'] = sum(stats['corloc_avg']) / float(
                    len(stats['corloc_avg']))

            with open(per_class_perf, 'w') as corloc:
                yaml.dump(stats, corloc)
            
            return stats['corloc_avg'], len([stats['corloc'][k] for k in stats['corloc']])

        else:
            with open(per_class_perf, 'w') as corloc:
                stats = {
                    'dataset': self.args.dataset,
                    'split': self.split,
                    'method': self.args.method,
                    'task': self.args.task,
                    'corloc_avg': loc_score.item(),
                    'corloc': {
                        cl_name: loc_score.item()
                    }
                }
                yaml.dump(stats, corloc)
            return None, None
                
    