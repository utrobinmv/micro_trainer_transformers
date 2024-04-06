import time
import os
import torch
import numpy as np
import random
import json
from json import JSONEncoder

def division(a,b):
    return a / b

def current_time_in_second(): 
    #Возвращает время в секундах
    return round(time.time())

def time_in_second_to_textdate(time_in_second):  
    #Преобразовывает секунды в текстовую дату
    local_time = time.localtime(time_in_second)
    str_time = time.strftime("%Y_%m_%d_%H-%M-%S", local_time)
    return str_time

def time_in_second_hms(time_in_second):
    #Преобразовывает секунды в текстовое время
    return time.strftime("%H:%M:%S", time.gmtime(time_in_second))

def set_seed(seed: int = 777, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)

def make_dirs(path,silent=False):
    try:
        os.makedirs(path)
    except OSError:
        if not silent:
            print ("Создать директорию %s не удалось" % path)
    else:
        if not silent:        
            print ("Успешно создана директория %s " % path)

def save_str_to_file(filename:str, str_text: str) -> None:
    '''
    Сохраняет строку в файл
    '''
    my_file = open(filename, "w")
    my_file.write(str_text)
    my_file.close()  

def in_jupyter_notebook():
    import __main__ as main
    return not hasattr(main, '__file__')

def save_useful_info(log_dir, model, params:dict, pl_params:dict):
    def remove_non_primitive_types(dictionary):
        primitive_types = (int, float, str, bool)

        for key in dictionary.keys():
            value = dictionary[key]
            if not any(isinstance(value, t) for t in primitive_types):
                dictionary[key] = str(dictionary[key])

        return dictionary
    
    save_str_to_file(f'{log_dir}/model.txt',str(model))

    save_params = params.copy()
    save_params = remove_non_primitive_types(save_params)
    
    with open(f'{log_dir}/params.json', 'w') as fp:
        json.dump(save_params, fp, indent=4)    

    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
                a = 1
                return str(o)

    with open(f'{log_dir}/pl_params.json', 'w') as fp:
        json.dump(pl_params, fp, indent=4, cls=CustomEncoder)    
        
    os.system(f"pip freeze > {log_dir}/requirements.txt")
    os.system(f"python -V > {log_dir}/python.txt")
    os.system(f"echo $VIRTUAL_ENV > {log_dir}/virtualenv.txt")
    os.system(f"uname -n > {log_dir}/hostname.txt")
    

def optimizer_to(optim, device):
    '''
    convert tesors optimizer to device
    '''
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
