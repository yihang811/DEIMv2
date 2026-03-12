"""
DEIMv2 Small Object Detection Training Script
Training script for small object detection with P2 feature and enhancement modules
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


def load_pretrained_weights_compatible(model, pretrained_path, strict=False):
    """
    Load pretrained weights with compatibility handling for new modules.
    
    Args:
        model: The model to load weights into
        pretrained_path: Path to pretrained checkpoint
        strict: Whether to strictly enforce that the keys match
    
    Strategy:
        - Load weights that match exactly (original parameters)
        - Initialize new parameters with Kaiming initialization
        - Skip parameters that don't exist in pretrained model
    """
    if not os.path.exists(pretrained_path):
        print(f"Pretrained weights not found at {pretrained_path}")
        return
    
    print(f"Loading pretrained weights from {pretrained_path}...")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint
    
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
                mismatched_keys.append(k)
        else:
            # Try to find corresponding key with different prefix
            found = False
            for mk in model_dict.keys():
                if mk.endswith(k) or k.endswith(mk.split('.')[-1]):
                    if v.shape == model_dict[mk].shape:
                        matched_dict[mk] = v
                        found = True
                        break
            if not found:
                new_keys.append(k)
    
    # 2. Load matched parameters
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)
    
    # 3. Print summary
    print(f"\n=== Pretrained Weights Loading Summary ===")
    print(f"Matched parameters: {len(matched_dict)}")
    print(f"Mismatched shape parameters: {len(mismatched_keys)}")
    print(f"New parameters (not loaded): {len(new_keys)}")
    
    if mismatched_keys:
        print(f"\nMismatched shape keys (first 10):")
        for k in mismatched_keys[:10]:
            print(f"  - {k}: pretrained {pretrained_dict[k].shape} vs model {model_dict[k].shape}")
    
    if new_keys:
        print(f"\nNew keys in pretrained (not used, first 10):")
        for k in new_keys[:10]:
            print(f"  - {k}")
    
    # 4. Initialize new modules with Kaiming initialization
    new_modules = []
    for name, param in model.named_parameters():
        if name not in matched_dict:
            new_modules.append(name.split('.')[0])
    
    new_modules = list(set(new_modules))
    if new_modules:
        print(f"\nNew modules initialized with Kaiming initialization:")
        for module_name in new_modules:
            print(f"  - {module_name}")
            # Apply Kaiming initialization to new modules
            for name, module in model.named_modules():
                if name.startswith(module_name):
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
                        if module.weight is not None:
                            nn.init.ones_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                        if module.running_mean is not None:
                            nn.init.zeros_(module.running_mean)
                        if module.running_var is not None:
                            nn.init.ones_(module.running_var)
                    elif isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
    
    print(f"\n=== Loading Complete ===\n")
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
    
    # Load pretrained weights with compatibility handling
    if args.pretrained and not args.resume and not args.tuning:
        solver.model = load_pretrained_weights_compatible(solver.model, args.pretrained)

    if args.test_only:
        solver.val()
    else:
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
                        help='path to pretrained weights (compatible loading)')
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
