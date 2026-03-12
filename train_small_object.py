"""
DEIMv2 Small Object Detection Training Script
Training script for small object detection with P2 feature and enhancement modules

Usage:
    # Train from scratch
    python train_small_object.py -c configs/deimv2/deimv2_dinov3_x_small_object.yml

    # Fine-tune from pretrained weights (compatible loading)
    python train_small_object.py -c configs/deimv2/deimv2_dinov3_x_small_object.yml -t path/to/pretrained.pth
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def load_pretrained_weights_compatible(model, pretrained_path, print_info=True):
    """
    Load pretrained weights with compatibility handling for new modules.
    
    This function handles the case when loading weights from a model with 
    different architecture (e.g., 3-scale to 4-scale feature pyramid).
    
    Args:
        model: The model to load weights into
        pretrained_path: Path to pretrained checkpoint
        print_info: Whether to print loading information
    
    Strategy:
        - Load weights that match exactly (original parameters)
        - Initialize new parameters with Kaiming initialization
        - Skip parameters that don't exist in pretrained model
    """
    if not os.path.exists(pretrained_path):
        print(f"Pretrained weights not found at {pretrained_path}")
        return model
    
    if print_info:
        print(f"Loading pretrained weights from {pretrained_path}...")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        pretrained_dict = checkpoint['ema']['module']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            new_pretrained_dict[k[7:]] = v
        else:
            new_pretrained_dict[k] = v
    pretrained_dict = new_pretrained_dict
    
    model_dict = model.state_dict()
    
    # 1. Filter out keys that don't match in shape or don't exist
    matched_dict = {}
    mismatched_keys = []
    new_keys = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                mismatched_keys.append((k, v.shape, model_dict[k].shape))
        else:
            new_keys.append(k)
    
    # 2. Load matched parameters
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)
    
    # 3. Print summary
    if print_info:
        print(f"\n{'='*50}")
        print(f"Pretrained Weights Loading Summary")
        print(f"{'='*50}")
        print(f"Matched parameters: {len(matched_dict)}")
        print(f"Mismatched shape: {len(mismatched_keys)}")
        print(f"Not in model: {len(new_keys)}")
        
        if mismatched_keys:
            print(f"\nMismatched shape keys (first 5):")
            for k, ps, ms in mismatched_keys[:5]:
                print(f"  {k}: pretrained {ps} vs model {ms}")
        
        if new_keys and len(new_keys) < 20:
            print(f"\nKeys not in model (first 10):")
            for k in new_keys[:10]:
                print(f"  {k}")
        
        # 4. Initialize new modules
        new_modules = set()
        for name in model_dict.keys():
            if name not in matched_dict:
                parts = name.split('.')
                for i in range(len(parts)):
                    new_modules.add('.'.join(parts[:i+1]))
        
        if new_modules:
            print(f"\nNew parameters initialized with Kaiming initialization:")
            for module_name in sorted(new_modules):
                if '.' not in module_name or module_name.endswith('weight') or module_name.endswith('bias'):
                    continue
                print(f"  {module_name}")
        
        print(f"{'='*50}\n")
    
    return model


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        # For compatible pretrained weights loading
        # We use the tuning mechanism built into DEIMv2
        if args.pretrained and not args.resume and not args.tuning:
            # Set tuning path - this will use BaseSolver.load_tuning_state
            # which handles compatible loading automatically
            cfg.tuning = args.pretrained
            solver.cfg.tuning = args.pretrained
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, 
                        default='configs/deimv2/deimv2_dinov3_x_small_object.yml')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-p', '--pretrained', type=str, 
                        default='',
                        help='path to pretrained weights (compatible loading for new architecture)')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
