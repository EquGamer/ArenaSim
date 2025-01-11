from omegaconf import DictConfig, OmegaConf
from typing import Dict

def omegaconf_to_dict(d: DictConfig)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret

def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print('')
        nesting += 4
        for k in val:
            print(nesting * ' ', end='')
            print(k, end=': ')
            print_dict(val[k], nesting, start=False)
    else:
        print(val)