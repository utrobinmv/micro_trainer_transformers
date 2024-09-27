from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

@dataclass
class TrainingArguments:
    framework = "pt"
    output_dir: str #Выходной каталог, в который будут записаны прогнозы модели и контрольные точки.
    overwrite_output_dir: bool = False #Перезаписать содержимое выходного каталога.

    do_train: bool = False #Запускать ли обучение.
    do_eval: bool = False #Следует ли запускать eval на наборе dev.
    do_predict: bool = False #Следует ли запускать прогнозы на тестовом наборе.
    evaluation_strategy: str = "no" #Union[IntervalStrategy, str] #Используемая стратегия оценки.
    prediction_loss_only: bool = False #При выполнении оценки и прогнозов возвращается только потеря.

    per_device_train_batch_size: int = 8 #Размер партии на ядро GPU/TPU/CPU для обучения.
    per_device_eval_batch_size: int = 8 #Размер партии на ядро GPU/TPU/CPU для оценки.

    gradient_accumulation_steps: int = 1 #Количество шагов обновления, которое нужно накопить перед выполнением прохода назад/обновления.
    eval_accumulation_steps: Optional[int] = None #Количество шагов прогнозирования, которое нужно накопить перед перемещением тензоров в ЦП.

    eval_delay: Optional[float] = 0 #Количество эпох или шагов ожидания, прежде чем можно будет выполнить первую оценку, в зависимости от стратегии оценки.

    learning_rate: float = 5e-5 #Начальная скорость обучения для AdamW.
    weight_decay: float = 0.0 #Снижение веса для AdamW, если мы применим некоторые.
    adam_beta1: float = 0.9 #Beta1 для оптимизатора AdamW
    adam_beta2: float = 0.999 #Beta2 для оптимизатора AdamW
    adam_epsilon: float = 1e-8 #Эпсилон для оптимизатора AdamW.
    max_grad_norm: float = 1.0 #Максимальная норма градиента.

    num_train_epochs: float = 3.0 #Общее количество тренировочных эпох для выполнения.
    max_steps: int = -1 #Если > 0: установите общее количество шагов обучения для выполнения. Переопределить num_train_epochs.

    lr_scheduler_type: str = "linear" #Union[SchedulerType, str] #Используемый тип планировщика.
    warmup_ratio: float = 0.0 #Линейный разогрев по отношению к разминке_отношения от общего количества шагов.
    warmup_steps: int = 0 #Линейная разминка по warmup_steps.

    log_level: Optional[str] = "passive" #Уровень журнала регистратора для использования на основном узле. 'info', 'warning', 'error' and 'critical', plus a 'passive' level, который ничего не устанавливает и позволяет приложению устанавливать уровень. По умолчанию 'passive'.

    log_level_replica: Optional[str] = "warning" #ровень журнала регистратора для использования на узлах-репликах. Те же варианты и значения по умолчанию, что и ``log_level``
    log_on_each_node: bool = True #При выполнении многоузлового распределенного обучения следует регистрировать один раз на каждом узле или только один раз на основном узле.
    logging_dir: Optional[str] = None #Каталог журнала Tensorboard.
    logging_strategy: str = "steps" #Union[IntervalStrategy, str] #Используемая стратегия ведения журнала.

    logging_first_step: bool = False #Зарегистрируйте первый global_step
    logging_steps: int = 500 #Регистрируйте каждые X шагов обновления.
    logging_nan_inf_filter: bool = True #Отфильтруйте потери nan и inf для регистрации.
    save_strategy: str = "steps" #Union[IntervalStrategy, str] #Стратегия сохранения контрольной точки.
    save_steps: int = 500 #Сохраняйте контрольную точку через каждые X шагов обновления.
    save_total_limit: Optional[int] = None #Ограничьте общее количество контрольных точек. Удаляет старые контрольные точки в output_dir. По умолчанию неограниченные контрольные точки.
    save_safetensors: Optional[bool] = False #Используйте сохранение и загрузку безопасных тензоров для дикторов состояния вместо стандартных torch.load и torch.save.
    save_on_each_node: bool = False #При многоузловом распределенном обучении сохранять ли модели и чекпоинты на каждом узле или только на основном
    no_cuda: bool = False #Не используйте CUDA, даже если она доступна
    use_mps_device: bool = False #Использовать ли устройство mps на базе чипа Apple Silicon.
    seed: int = 42 #Случайное семя, которое будет установлено в начале обучения.
    data_seed: Optional[int] = None #Случайное начальное число для использования с data samplers. 
    jit_mode_eval: bool = False #Следует ли использовать JIT-трассировку PyTorch для логического вывода
    use_ipex: bool = False #Use Intel extension for PyTorch when it is available, installation: https://github.com/intel/intel-extension-for-pytorch
    bf16: bool = False #Использовать ли точность bf16 (смешанную) вместо 32-битной. Требуется архитектура NVIDIA Ampere или выше или использование процессора (no_cuda). Это экспериментальный API, и он может измениться.
    fp16: bool = False #Использовать ли точность fp16 (смешанную) вместо 32-битной
    fp16_opt_level: str = "O1" #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html"
    half_precision_backend: str = "auto" #Серверная часть, которая будет использоваться для половинной точности. ["auto", "cuda_amp", "apex", "cpu_amp"]
    bf16_full_eval: bool = False #Использовать ли полную оценку bfloat16 вместо 32-битной. Это экспериментальный API, и он может измениться.
    fp16_full_eval: bool = False #Использовать ли полную оценку float16 вместо 32-битной
    tf32: Optional[bool] = None #Включить ли режим tf32, доступный в Ampere и более новых архитектурах GPU. Это экспериментальный API, и он может измениться.
    local_rank: int = -1 #Для распределенного обучения: local_rank
    xpu_backend: Optional[str] = None #Серверная часть, которая будет использоваться для распределенного обучения на Intel XPU. ["mpi", "ccl", "gloo"]
    tpu_num_cores: Optional[int] = None #TPU: количество ядер TPU (автоматически передается сценарием запуска)
    tpu_metrics_debug: bool = False #Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: следует ли печатать метрики отладки
    debug: str = "" #Включать или нет режим отладки. Current options: `underflow_overflow` (Обнаружение недополнения и переполнения в активациях и весах), `tpu_metrics_debug` (print debug metrics on TPU).

    dataloader_drop_last: bool = False #Отбросить последнюю незавершенную партию, если она не делится на размер партии.
    eval_steps: Optional[int] = None #Запускайте оценку через каждые X шагов.
    dataloader_num_workers: int = 0 #Количество подпроцессов, используемых для загрузки данных (только PyTorch). 0 означает, что данные будут загружены в основном процессе.

    past_index: int = -1 #Если >=0, использует соответствующую часть вывода в качестве прошлого состояния для следующего шага.

    run_name: Optional[str] = None #Необязательный дескриптор запуска. В частности, используется для регистрации wandb.
    disable_tqdm: Optional[bool] = None #Следует ли отключать индикаторы выполнения tqdm.

    remove_unused_columns: Optional[bool] = True #Удалите столбцы, не требуемые моделью, при использовании nlp.Dataset.
    label_names: Optional[List[str]] = None #Список ключей в вашем словаре входных данных, соответствующих меткам.

    load_best_model_at_end: Optional[bool] = False #Следует ли загружать лучшую модель, найденную во время обучения, в конце обучения.
    metric_for_best_model: Optional[str] = None #Метрика, используемая для сравнения двух разных моделей.
    greater_is_better: Optional[bool] = None #Следует ли максимизировать metric_for_best_model или нет.
    ignore_data_skip: bool = False #При возобновлении обучения, следует ли пропускать первые эпохи и партии, чтобы получить те же данные обучения.
    sharded_ddp: str = "" #Следует ли использовать сегментированное обучение DDP (только в распределенном обучении). The base option should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or `zero_dp_3` with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.
    fsdp: str = "" #Следует ли использовать обучение PyTorch Fully Sharded Data Parallel (FSDP) (только в распределенном обучении). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard auto_wrap` or `shard_grad_op auto_wrap`.
    fsdp_min_num_params: int = 0 #Этот параметр устарел. Минимальное количество параметров FSDP для автоматической упаковки по умолчанию. (полезно только при передаче поля `fsdp`).
    fsdp_config: Optional[str] = None #Конфигурация для использования с FSDP (Pytorch Fully Sharded Data Parallel). The  value is either a fsdp json config file (e.g., `fsdp_config.json`) or an already loaded  json file as `dict`.
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = None #Этот параметр устарел. Имя класса слоя Transformer (с учетом регистра) для переноса, e.g, `BertLayer`, `GPTJBlock`, `T5Block` .... (useful only when `fsdp` flag is passed).
    deepspeed: Optional[str] = None #Включите DeepSpeed и передайте путь к файлу конфигурации DeepSpeed JSON (например, ds_config.json) или уже загруженному JSON-файлу в виде dict"
    label_smoothing_factor: float = 0.0 #Применяемый эпсилон сглаживания метки (ноль означает отсутствие сглаживания метки).

    default_optim = "adamw_hf"
    # XXX: включить, когда выйдет pytorch == 2.0.1 - мы хотим дать ему время, чтобы разобраться со всеми ошибками
    # if is_torch_available() and version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.1.0"):
    #     default_optim = "adamw_torch_fused"
    # and update the doc above to:
    # optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch_fused"` (for torch<2.1.0 `"adamw_hf"`):

    optim: str = default_optim #Union[OptimizerNames, str] #Оптимизатор для использования.
    optim_args: Optional[str] = None #Необязательные аргументы для предоставления оптимизатору.
    adafactor: bool = False #Стоит ли заменять AdamW на Adafactor.
    group_by_length: bool = False #Группировать ли образцы примерно одинаковой длины вместе при группировании.
    
    length_column_name: Optional[str] = "length" #Имя столбца с предварительно вычисленной длиной для использования при группировке по длине.

    report_to: Optional[List[str]] = None #Список интеграций, в которые следует отправлять отчеты о результатах и журналах.
    ddp_find_unused_parameters: Optional[bool] = None #При использовании распределенного обучения значение флага `find_unused_parameters` passed to `DistributedDataParallel`.
    ddp_bucket_cap_mb: Optional[int] = None #При использовании распределенного обучения значение флага `bucket_cap_mb` passed to `DistributedDataParallel`.
    dataloader_pin_memory: bool = True #Следует ли закреплять память для DataLoader.
    skip_memory_metrics: bool = True #Следует ли пропускать добавление отчетов профилировщика памяти в метрики.
    use_legacy_prediction_loop: bool = False #Следует ли использовать устаревший метод предсказания_цикла в Trainer.
    push_to_hub: bool = False #Следует ли загружать обученную модель в hub моделей после обучения.
    resume_from_checkpoint: Optional[str] = None #Путь к папке с допустимой контрольной точкой для вашей модели.

    hub_model_id: Optional[str] = None #Имя репозитория для синхронизации с локальным `output_dir`.
    hub_strategy: str = "every_save" #Union[HubStrategy, str] #Стратегия хаба, которую следует использовать, когда `--push_to_hub` is activated.
    hub_token: Optional[str] = None #Токен, который будет использоваться для отправки Model Hub.
    hub_private_repo: bool = False #Является ли репозиторий модели частным или нет.
    gradient_checkpointing: bool = False #If True, используйте контрольные точки градиента, чтобы сэкономить память за счет более медленного обратного прохода.
    include_inputs_for_metrics: bool = False #Будут ли вводы переданы в `compute_metrics` function.
    
    auto_find_batch_size: bool = False #Следует ли автоматически уменьшать размер пакета вдвое и перезапускать цикл обучения каждый раз, когда достигается нехватка памяти CUDA.
    full_determinism: bool = False #Следует ли вызывать enable_full_determinism вместо set_seed для воспроизводимости в распределенном обучении. Важно: это негативно скажется на производительности, поэтому используйте его только для отладки.
    ray_scope: Optional[str] = "last" #Область, используемая при выполнении поиска гиперпараметров с помощью Ray. По умолчанию будет использоваться «последний». Затем Рэй использует последнюю контрольную точку всех испытаний, сравнивает их и выбирает лучшее. Однако возможны и другие варианты. См. документацию Рэя (https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial) for more options.
    ddp_timeout: Optional[int] = 1800 #Переопределяет время ожидания по умолчанию для распределенного обучения (значение должно быть указано в секундах).
    torch_compile: bool = False #If set to `True`, модель будет завернута в `torch.compile`.
    torch_compile_backend: Optional[str] = None #Какой бэкэнд использовать с `torch.compile`, передача которого вызовет компиляцию модели.
    torch_compile_mode: Optional[str] = None #Какой режим использовать с `torch.compile`, передача которого вызовет компиляцию модели.

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    sortish_sampler: bool = False #Использовать SortishSampler или нет.
    predict_with_generate: bool = False #Следует ли использовать generate для расчета генеративных метрик (ROUGE, BLEU).
    generation_max_length: Optional[int] = None #`max_length` для использования в каждом цикле оценки, когда `predict_with_generate=True`. По умолчанию будет установлено значение `max_length` конфигурации модели.
    generation_num_beams: Optional[int] = None #`num_beams` для использования в каждом цикле оценки, когда `predict_with_generate=True`. По умолчанию будет установлено значение num_beams конфигурации модели.
    generation_config: Optional[Union[str, Dict]] = None #Optional[Union[str, Path, GenerationConfig]] #Идентификатор модели, путь к файлу или URL-адрес, указывающий на json-файл GenerationConfig для использования во время прогнозирования.
    