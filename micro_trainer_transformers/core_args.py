from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .utils import current_time_in_second, time_in_second_to_textdate
from .arguments import TrainingArguments

@dataclass
class TrainigParameters:
    """Класс для параметров тренировки по умолчанию"""
    wandb: bool = False
    max_train_steps: Optional[int] = None
    max_train_epochs: Optional[int] = None #При None max_train_epochs = max_train_steps / val_check_interval
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = 'steps' #'steps' or 'epoch'. Делать валидацию либо каждую эпоху или через val_check_interval шагов
    warmup_steps: Optional[int] = None
    val_check_interval: int = 0 #Использовать всегда, если evaluation_strategy, устанавливать как len(ds) / (batch_size * evaluation_strategy)
    
    num_workers: int = 4
    seed: int = 42 #seed
    collate_fn: Optional[Any] = None
    learning_rate: float = 0.001
    weight_decay: float = 1e-2
    eps: float = 1e-8
    max_grad_norm: float = 1.0 #Максимальная норма градиента.

    #data
    batch_size: int = 2
    #evaluation_strategy: int = 1 #Странная строка, вероятно здесь должна быть другая переменная?
    data_train_shuffle: bool = False
    data_streaming_train: bool = False
    data_streaming_train_iter_replace: bool = True #Не завершать эпоху по окончанию итеративного датасета, а перезапускать заново!
    data_streaming_valid: bool = False
    limit_val_batches: int or float or None = None

    #lr_scheduler_type
    lr_scheduler_type: str = 'OneCycleLR' #"linear","cosine","cosine_with_restarts"
    lr_scheduler_interval: str = 'step' #step or epoch 
            #Интервал для lr_scheduler step
            # "polynomial","constant","constant_with_warmup","inverse_sqrt"
            # "OneCycleLR"
            #Интервал epoch
            # "constant","ReduceLROnPlateau", 
    metric_monitor_name: str = 'valid_loss' #Монитор метрики в сохранении чекпоинтов и ReduceLROnPlateau
    metric_monitor_mode: str = 'min' #min означает, чем метрика меньше тем лучше
    lr_scheduler_kwargs: Optional[Dict] = None #Дополнительные опции lr_scheduler_kwargs, используются не для всех scheduler
    
    #save_checkpoints
    early_stopping_patience: Optional[int] = None #Если значение > 0, то используется early_stopping через каждые N неудачных эпох
    save_total_limit: int = 3 #Ограничьте общее количество контрольных точек. Удаляет старые контрольные точки в output_dir. По умолчанию неограниченные контрольные точки.

    #log
    log_every_n_steps: int = 20
    
    #floating point
    tf32: Optional[bool] = None #Включить ли режим tf32, доступный в Ampere и более новых архитектурах GPU. Это экспериментальный API, и он может измениться.
    bf16: bool = False #Использовать ли точность bf16 (смешанную) вместо 32-битной. Требуется архитектура NVIDIA Ampere или выше или использование процессора (no_cuda). Это экспериментальный API, и он может измениться.
    fp16: bool = False #Использовать ли точность fp16 (смешанную) вместо 32-битной
    torch_compile: bool = False #If set to `True`, модель будет завернута в `torch.compile`.

    #project    
    project_name: str = 'debug_classificate'
    model_master_name: str = 'linear_simple'
    model_comment: str = ''
    root_path_checkpoint = '/checkpoints'

    #local trainer
    local_trainer: bool | str = False  # False True 'sentence'
    
    def __init__(self):
        
        self.accum_param()
        self.change_dir()

    def accum_param(self):

        #assert self.val_check_interval > 0 #Необходимо всегда установить val_check_interval

        #TO DO
        ########evaluation_strategy=<IntervalStrategy.STEPS: 'steps' ???

        if self.evaluation_strategy == 'epoch':
            if self.max_train_epochs is None:
                self.max_train_epochs = round(self.max_train_steps / self.val_check_interval)
                self.max_train_steps = self.val_check_interval * self.max_train_epochs #Пересчитаем до полного числа эпох
        elif self.evaluation_strategy == 'steps':
            # Считаю эту строку неверной, так как количество шагов определяется от размера датасета
            # if self.max_train_steps is None or self.max_train_steps == -1:
            #     self.max_train_steps = self.val_check_interval * self.max_train_epochs #Пересчитаем до полного числа эпох
            pass
        pass


    def change_dir(self):
        #not edit
        self.run_time: int = current_time_in_second()
        self.version: str = time_in_second_to_textdate(self.run_time)
        self.model_name: str = self.model_master_name + '_'+ self.version # + ('_debug' if self.debug else '')
        
        self.path_checkpoint: str = self.root_path_checkpoint + '/' + self.project_name + '/' + \
            self.version + '_' + \
            self.model_master_name  + '/' #+ ('_debug' if self.debug else '') + '/' 
            
        self.path_log: str = self.path_checkpoint + 'logs/'
        self.path_best_model: str = self.root_path_checkpoint + '/save_models/'
        pass

    def from_hf_training_arguments(self, project_name: str, model_master_name: str, ta: TrainingArguments):
        self.project_name = project_name
        self.model_master_name = model_master_name

        #ignore parameters
        _ = ta.output_dir
        _ = ta.overwrite_output_dir
        _ = ta.per_device_eval_batch_size
        _ = ta.save_strategy #save_strategy должно быть одинаково с evaluation_strategy (дублирование параметров)
        _ = ta.save_steps #у меня анализируется данные из eval_steps
        
        _ = ta.load_best_model_at_end #Не реализовано
        
        assert ta.eval_steps == ta.save_steps #У меня это один и тот же параметр, значит данные должны быть равны
        
        self.max_grad_norm = ta.max_grad_norm
        self.evaluation_strategy = ta.evaluation_strategy
        self.learning_rate = ta.learning_rate
        self.max_train_steps = ta.max_steps
        self.max_train_epochs = ta.num_train_epochs
        self.batch_size = ta.per_device_train_batch_size
        self.gradient_accumulation_steps = ta.gradient_accumulation_steps
        self.val_check_interval = ta.eval_steps
        if ta.lr_scheduler_type:
            self.lr_scheduler_type = ta.lr_scheduler_type._value_
        self.seed = ta.seed

        #Убрал так как метрика по умолчанию 'loss', а у меня 'valid_loss'
        # if ta.metric_for_best_model: 
        #     self.metric_monitor_name = ta.metric_for_best_model

        self.tf32 = ta.tf32
        self.bf16 = ta.bf16
        self.fp16 = ta.fp16
        self.torch_compile = ta.torch_compile
        
        if ta.warmup_steps > 0:
            self.warmup_steps = ta.warmup_steps
        elif ta.warmup_ratio > 0:
            self.warmup_steps = int(ta.max_steps * ta.warmup_ratio)

        if ta.greater_is_better:
            if ta.greater_is_better == True:
                self.metric_monitor_mode = 'max'
            else:
                self.metric_monitor_mode = 'min'
        
        self.log_every_n_steps = ta.logging_steps
        if ta.save_total_limit and ta.save_total_limit > 0:
            self.save_total_limit = ta.save_total_limit

        self.accum_param()
        self.change_dir()
        
        assert self.val_check_interval > 0 #Необходимо всегда установить val_check_interval
        
        pass