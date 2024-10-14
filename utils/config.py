import yaml
from easydict import EasyDict
import os

def merge_new_config(config, new_config):
    for key,val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'],'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    
    return config
        
def cfg_from_yaml_file(yaml_file):
    config = EasyDict()
    with open(yaml_file, 'r') as f:
        try:
            yaml_cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        except:
            yaml_cfg = EasyDict(yaml.load(f))
    
    merge_new_config(config=config, new_config=yaml_cfg)
    return config

def get_config(args):
    config = cfg_from_yaml_file(args.config)
    save_experiment_config(args, config)
    return config

def save_experiment_config(args, config):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))