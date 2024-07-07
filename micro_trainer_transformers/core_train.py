from typing import Any, Callable, Dict, NewType, Union, List, Optional, Tuple

import gc
from tqdm.auto import tqdm

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast

import datasets
from datasets.iterable_dataset import IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

from micro_trainer_transformers import TrainigParameters
from .utils import time_in_second_hms, make_dirs, set_seed, save_useful_info
from .utils import in_jupyter_notebook
from .core_optim import UniversalOptim

class UniversalTrainingModule(pl.LightningModule, UniversalOptim):
    def __init__(
        self,
        model: nn.Module,
        args: TrainigParameters, 
        train_dataset: datasets.Dataset,
        eval_dataset: datasets.Dataset,
        data_collator: Any = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        compute_metrics: Any = None,
        evaluator: Any = None,
        loss: Any = None,
        
    ) -> None:
        super().__init__()

        self.m_trainer = None

        self.optimizers = optimizers
        self.evaluator = evaluator

        self.training_params = args
        
        self.dataset_train = train_dataset
        self.dataset_valid = eval_dataset
        
        if self.training_params.data_streaming_train:
            self.dataset_train_size = 0
            self.dataset_train_size_count = 0 #Подсчет размера итеративного датасета
            self.dataset_train_iter = iter(self.dataset_train)
        else:
            if isinstance(self.dataset_train, IterableDataset):
                #train_dataloader.dataset.set_epoch(args.current_epoch)
                self.dataset_train_size = self.training_params.batch_size * self.training_params.val_check_interval
                self._dataloader_current_epoch = 1 #Это не настоящий номер эпохи, это скорее то, сколько раз вызывается даталоадер
                assert self.dataset_train.n_shards >= self.training_params.num_workers
                
                if self.training_params.data_train_shuffle:
                    self.training_params.data_train_shuffle = False
                    print(' === Change option data_train_shuffle, for IterableDataset not use option shuffle.')
                
            else:
                self.dataset_train_size = len(self.dataset_train)

        if self.training_params.data_streaming_valid:
            self.dataset_valid_size = 0
            self.dataset_valid_size_count = 0 #Подсчет размера итеративного датасета
            self.dataset_valid_iter = iter(self.dataset_valid)
        else:
            if isinstance(self.dataset_valid, IterableDataset):
                self.dataset_valid_size = 0
            elif isinstance(self.dataset_valid, Dataset):
                self.dataset_valid_size = len(self.dataset_valid)
            else:
                self.dataset_valid_size = 0
        self.dataloader_valid = None
        self.dataset_total_batches = 0
        
        #for resume checkpoint
        self.dataset_start_epoch_data_iter = 0
        self.pbar_train_restore_n = 0
        self.checkpoint_callback = None
        
        self.model = model
        self.data_collator = data_collator
        self.compute_metrics_fn = compute_metrics
        
        self.loss_fn = loss
        
        self.pbar_train = None
        if self.training_params.local_trainer == False:
            self.pbar_train = tqdm(desc='Total training model',leave=True,position=2) 
        
        self.train_mode_start = False
        self.save_first_valid_batch = False
        self.first_valid_batch = None

        #Аккумулирование шагов лоса, чтобы в конце эпохи получить среднюю метрику
        self.epoch_train_logs = {}
        self.epoch_valid_logs = {}
        self.epoch_valid_list_dict_result = []

        self.validate_labels = []
        self.validate_predictions = []

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        pass

    def on_fit_start(self):
        #print('on fit start')
        #self.current_training_step = iter(range(self.training_params.max_train_steps))
        #self.pbar_train = iter(tqdm(range(self.training_params.max_train_steps),desc='Training model',leave=True))
        
        self.train_mode_start = True
        
        if self.pbar_train is not None:
            self.pbar_train.clear()
            if self.training_params.max_train_steps is None or self.training_params.max_train_steps == -1:
                self.pbar_train.reset(total=self._trainer.estimated_stepping_batches)
            else:
                print('max_train_steps:',self.training_params.max_train_steps)
                self.pbar_train.reset(total=self.training_params.max_train_steps)
                
            if self.pbar_train_restore_n > 0: #restore from checkpoint
                self.pbar_train.n = self.pbar_train_restore_n
                self.pbar_train_restore_n = 0
        
        pass 

    def on_train_start(self):
        # self.validate_labels.clear()
        # self.validate_predictions.clear()
       
        pass
    
    def train_dataloader(self):
        #print(' === reload train dataloader...')
        if self.training_params.data_streaming_train:
            self.dataset_start_epoch_data_iter = self.dataset_train_size_count
            
            buffer_size = self.training_params.batch_size * self.training_params.val_check_interval * self.training_params.gradient_accumulation_steps
            self.data_buffer_train = []
            for _ in tqdm(range(buffer_size), desc="Reload train dataloader...",leave=False,position=3):
                #self.pbar_train.display()
                if self.dataset_train_size_count % (self.training_params.batch_size * self.training_params.gradient_accumulation_steps)  == 0:
                    if self.pbar_train is not None:
                        #self.pbar_train.refresh()
                        self.pbar_train.update(n=0)
                        self.pbar_train.display()
                    
                if buffer_size <= 0:
                    break
                item = next(self.dataset_train_iter, "end")
                if item == "end": #Нашли конец датасета
                    self.dataset_train_size = self.dataset_train_size_count
                    #Перезапускаем чтение датасета сначала
                    self.dataset_train_iter = iter(self.dataset_train)
                    self.dataset_train_size_count = 0
                    if self.training_params.data_streaming_train_iter_replace:
                        #Считываем с начала первый элемент
                        item = next(self.dataset_train_iter)
                    else:
                        break
                self.data_buffer_train.append(item)
                buffer_size -= 1
                self.dataset_train_size_count += 1
                
            data_buffer = self.data_buffer_train
            
        else:
            if isinstance(self.dataset_train, IterableDataset):
                self.dataset_train.set_epoch(self._dataloader_current_epoch)
                self._dataloader_current_epoch += 1
                #buffer_size = self.training_params.batch_size * self.training_params.val_check_interval * self.training_params.gradient_accumulation_steps
                
            data_buffer = self.dataset_train 
            
        return DataLoader(data_buffer, 
            num_workers=self.training_params.num_workers,
            shuffle=self.training_params.data_train_shuffle, 
            collate_fn = self.data_collator if self.data_collator else None, 
            batch_size = self.training_params.batch_size, 
            drop_last=False, 
            pin_memory=True)

    def val_dataloader(self):
        if self.dataloader_valid is None:
            #print(' === load valid dataloader...')
            
            if self.training_params.data_streaming_valid:
                pbar = tqdm(desc='Load valid dataloader',leave=False)
                if not self.training_params.limit_val_batches is None:
                    buffer_size = self.training_params.batch_size * self.training_params.val_check_interval
                    pbar.reset(total=buffer_size)
                else:
                    buffer_size = float("inf")
                    
                self.data_buffer_valid = []
                
                while True:
                    pbar.update()
                    if buffer_size <= 0:
                        break
                    item = next(self.dataset_valid_iter, "end")
                    if item == "end": #Нашли конец датасета
                        self.dataset_valid_size = self.dataset_valid_size_count
                        self.dataset_valid_size_count = 0
                        break
                    self.data_buffer_valid.append(item)
                    buffer_size -= 1
                    self.dataset_valid_size_count += 1
                    
                data_buffer = self.data_buffer_valid
                
            else:
                data_buffer = self.dataset_valid 
        
            self.dataloader_valid = DataLoader(data_buffer,
                num_workers=self.training_params.num_workers,
                shuffle=False, 
                collate_fn = self.data_collator if self.data_collator else None, 
                batch_size = self.training_params.batch_size,
                drop_last=False,
                pin_memory=True)
        
        return self.dataloader_valid

    def test_dataloader(self):
        return None

    def configure_optimizers(self):
        return super().configure_optimizers_optim(self.optimizers)

    def on_load_checkpoint(self, checkpoint):
        self.dataset_start_epoch_data_iter = checkpoint['dataset_start_epoch_data_iter']
        self.pbar_train_restore_n = checkpoint['pbar_train_n']
        self.dataset_train_size = checkpoint['dataset_train_size']

        if self.training_params.data_streaming_train and self.dataset_start_epoch_data_iter > 0:
            for _ in tqdm(range(self.dataset_start_epoch_data_iter), desc="Restore train dataloader...",leave=False,position=3):
                _ = next(self.dataset_train_iter)
            self.dataset_train_size_count = self.dataset_start_epoch_data_iter
            
        pass

    def on_save_checkpoint(self, checkpoint):
        checkpoint['dataset_start_epoch_data_iter'] = self.dataset_start_epoch_data_iter
        if self.pbar_train is not None:
            checkpoint['pbar_train_n'] = self.pbar_train.n
        else:
            checkpoint['pbar_train_n'] = 0
        checkpoint['dataset_train_size'] = self.dataset_train_size

        pass

        
    def on_train_batch_start(self, batch, batch_idx):
        
        if batch_idx % self.training_params.gradient_accumulation_steps == 0 and self.train_mode_start:
            # if self.pbar_train.total == self.pbar_train.n:
            #     self.pbar_train.total += 1

            if self.pbar_train is not None:
                self.pbar_train.update()
                self.pbar_train.display()
                
                t1 = self.pbar_train.start_t
                t2 = self.pbar_train.last_print_t
                t = t2 - t1
                t_prognoz = t / self.pbar_train.n * self.pbar_train.total
                
                time_total = time_in_second_hms(t_prognoz)
                time_str = time_in_second_hms(t_prognoz - t)
                
                desc_str = f'Total train model [{time_total}/{time_str}]'
                
                #Подсчет эпох в зависимости от датасета
                if self.dataset_total_batches == 0:
                    if self.dataset_train_size > 0:
                        self.dataset_total_batches = self.dataset_train_size / (self.training_params.batch_size * self.training_params.gradient_accumulation_steps)
                else:                        
                    data_epoch = self.m_trainer.global_step / self.dataset_total_batches
                    desc_str += f'. Epoch {data_epoch:.2f}'
                    
                self.pbar_train.set_description(desc_str)

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     print(len(outputs),batch_idx)
        
    #     pass        

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
                print('get lr:', float(param_group['lr']))
                return float(param_group['lr'])
         
    def common_step(self, batch: Dict[str, torch.LongTensor], batch_idx: int, valid: bool = False) -> torch.Tensor:
               
        if not self.loss_fn:            
            preds = self.model(**batch)
            loss = preds.loss
            
            if valid:
                if self.compute_metrics_fn:
                    self.validate_labels.extend(list(batch['labels'].detach().cpu()))
                    self.validate_predictions.extend(list(preds.logits.argmax(2).detach().cpu()))
            
        else:
            x = batch['image']
            y = batch['label']

            pred = self.model(x)
            
            loss = self.loss_fn(pred, y)
        
        return loss
        
    def training_step(
        self, batch: Dict[str, torch.LongTensor], batch_idx: int,
    ) -> torch.Tensor:
        
        if isinstance(batch, dict):
            batch_size = len(batch[list(batch.keys())[0]])
        else:
            batch_size = len(batch["labels"])
        
        loss = self.common_step(batch, batch_idx)
        
        self.log("training_loss", loss.item(), on_step=True, 
                 on_epoch=True, prog_bar=False, 
                 batch_size=batch_size)
        
        if 'training_loss' not in self.epoch_train_logs:
            self.epoch_train_logs['training_loss'] = 0.0
            self.epoch_train_logs['training_items'] = 0
        self.epoch_train_logs['training_loss'] += loss.item()
        self.epoch_train_logs['training_items'] += 1

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], batch_idx: int,
    ) -> torch.Tensor:
        
        if isinstance(batch, dict):
            batch_size = len(batch[list(batch.keys())[0]])
        else:
            batch_size = len(batch["labels"])
            
        if self.save_first_valid_batch:
           self.first_valid_batch = batch
           self.save_first_valid_batch = False
            
        loss = self.common_step(batch, batch_idx, valid = True)
        
        self.log("valid_loss", loss.item(), on_step=False, 
                 on_epoch=True, prog_bar=True, 
                 batch_size=batch_size)

        if 'valid_loss' not in self.epoch_valid_logs:
            self.epoch_valid_logs['valid_loss'] = 0.0
            self.epoch_valid_logs['valid_items'] = 0
        self.epoch_valid_logs['valid_loss'] += loss.item()
        self.epoch_valid_logs['valid_items'] += 1


    def on_train_epoch_start(self):
        self.epoch_train_logs.clear()

        #cache clear
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        self.validate_labels.clear()
        self.validate_predictions.clear()
        self.epoch_valid_logs.clear()
        
        #cache clear
        # gc.collect()
        torch.cuda.empty_cache()

        pass

    def on_validation_epoch_end(self):

        #Соберем логи по тренировке и валидации
        #current_step = self.trainer.num_training_batches
        current_step: int = self.m_trainer.global_step
        dict_result = {}
        dict_result['step'] = current_step

        if 'training_loss' in self.epoch_train_logs:
            self.epoch_train_logs['training_loss'] = self.epoch_train_logs['training_loss'] / self.epoch_train_logs['training_items']
            self.epoch_train_logs.pop('training_items')
            dict_result.update(self.epoch_train_logs)

        if 'training_loss' not in dict_result.keys():
            dict_result['training_loss'] = 0.0

        if 'valid_loss' in self.epoch_valid_logs:
            self.epoch_valid_logs['valid_loss'] = self.epoch_valid_logs['valid_loss'] / self.epoch_valid_logs['valid_items']
            self.epoch_valid_logs.pop('valid_items')
            dict_result.update(self.epoch_valid_logs)

        if self.compute_metrics_fn:            
            
            predictions = self.validate_predictions
            #predictions = [value.argmax(1) for value in predictions]


            result_dict = self.compute_metrics_fn(
                (pad_sequence(predictions, batch_first=True, padding_value=-100), 
                 pad_sequence(self.validate_labels, batch_first=True, padding_value=-100))
                )
            #result_dict = self.compute_metrics_fn((torch.vstack(self.validate_predictions), torch.vstack(self.validate_labels)))

            if len(result_dict.keys()) > 0:
                dict_result.update(result_dict)
                self.log_dict(result_dict, on_step=False, on_epoch=True, prog_bar=False)

        if len(dict_result.keys()) > 0:
            self.epoch_valid_list_dict_result.append(dict_result)    
            if in_jupyter_notebook():
                from IPython.display import display, HTML
                pd.set_option('display.max_rows', None)
                df = pd.DataFrame(self.epoch_valid_list_dict_result)
                display(df)
                del df
            else:
                print(self.epoch_valid_list_dict_result[-1])

        #cache clear
        # gc.collect()
        torch.cuda.empty_cache()


    def create_trainer(self, mode='train'):

        set_seed(self.training_params.seed)
        
        if self.training_params.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        
        use_pl_trainer = not self.training_params.local_trainer

        loggers = []
        
        make_dirs(self.training_params.path_log + 'wandb/')
            
        if use_pl_trainer:

            loggers.append(pl.loggers.TensorBoardLogger(save_dir = self.training_params.path_log + 'tensorboard',
                                                name = None,
                                                version = '',
                                                ))
            loggers.append(pl.loggers.CSVLogger(save_dir = self.training_params.path_log + 'csv_log',
                                                name = None,
                                                version = '',
                                                ))
            
            if self.training_params.wandb and mode == 'train':
                
                wandb_logger = pl.loggers.WandbLogger(name = self.training_params.model_name,
                                                    project = self.training_params.project_name,
                                                    version = self.training_params.version, 
                                                    save_dir=self.training_params.path_log,
                                                    log_model=True)
                
                loggers.append(wandb_logger)
                
                #log gradients, parameter histogram and model topology (100 steps by default)
                wandb_logger.watch(self.model, log="all", log_freq=100)


            self.checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=self.training_params.metric_monitor_name, 
                                                        save_top_k=self.training_params.save_total_limit, 
                                                        save_last=True, 
                                                        dirpath = self.training_params.path_checkpoint,
                                                        filename = '{epoch}_{step}-{'+f'{self.training_params.metric_monitor_name}'+':.6f}',
                                                        mode=self.training_params.metric_monitor_mode,
                                                        )
        if use_pl_trainer:

            # plugins: list = []
            # if self.training_params.fp16:
            #     plugins.append(pl.plugins.precision.NativeMixedPrecisionPlugin(16,'cuda',GradScaler()))

            callbacks = []
            callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
            callbacks.append(pl.callbacks.TQDMProgressBar())
            callbacks.append(pl.callbacks.ModelSummary(max_depth=2))
            # callbacks.append(pl.callbacks.OnExceptionCheckpoint(dirpath=self.training_params.path_checkpoint,
            #                                                     filename='exception'))
            callbacks.append(self.checkpoint_callback)
            
            if self.training_params.evaluation_strategy == 'epoch':
                if self.training_params.early_stopping_patience:
                    callbacks.append(pl.callbacks.EarlyStopping(patience=self.training_params.early_stopping_patience,min_delta=0.001,
                                                                mode=self.training_params.metric_monitor_mode,
                                                                monitor=self.training_params.metric_monitor_name,
                                                                verbose=True))
                
        if self.training_params.evaluation_strategy == 'steps':
            if use_pl_trainer:
                val_check_interval = (self.training_params.val_check_interval)*self.training_params.gradient_accumulation_steps
            else:
                val_check_interval = self.training_params.val_check_interval
            check_val_every_n_epoch=None
        else: #if self.training_params.evaluation_strategy == 'epoch':
            val_check_interval = None
            check_val_every_n_epoch=1
        
        max_train_kwargs = {}
        if self.training_params.max_train_steps is None or self.training_params.max_train_steps == -1:
            if self.training_params.max_train_epochs is not None and self.training_params.max_train_epochs > 0 and int(self.training_params.max_train_epochs) != self.training_params.max_train_epochs:

                one_epoch_steps: float = (len(self.train_dataloader()) / self.training_params.gradient_accumulation_steps)
                self.training_params.max_train_epochs = self.training_params.max_train_epochs
                self.training_params.max_train_steps = int(one_epoch_steps * self.training_params.max_train_epochs)
                max_train_kwargs: dict[str, float] = {'max_steps': self.training_params.max_train_steps}
            else:
                max_train_steps = None
                max_train_kwargs = {'max_epochs': self.training_params.max_train_epochs}
        else:
            if use_pl_trainer:
                #Какой то косяк в pytorch lightning Приходится добавлять несколько заключительных шагов
                max_train_steps = self.training_params.max_train_steps + (self.training_params.gradient_accumulation_steps * 2)
            else:
                max_train_steps = self.training_params.max_train_steps
            max_train_kwargs = {'max_steps': max_train_steps}

        precision = 32
        if self.training_params.bf16:
            precision = "bf16"
        elif self.training_params.fp16:
            precision = 16

        if self.training_params.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        if self.training_params.torch_compile:
            print("compiling the model... (takes a ~minute)")
            #unoptimized_model = model
            self.model = torch.compile(self.model) # requires PyTorch 2.0            

                        
        if use_pl_trainer:

            #, max_steps=max_train_steps
            self.m_trainer = pl.Trainer(accelerator="auto", devices="auto", 
                    **max_train_kwargs,
                    precision=precision,
                    val_check_interval=val_check_interval,
                    accumulate_grad_batches=self.training_params.gradient_accumulation_steps,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    default_root_dir=self.training_params.path_best_model,
                    logger=loggers,
                    gradient_clip_val=self.training_params.max_grad_norm,
                    callbacks=callbacks,
                    # plugins=plugins,
                    reload_dataloaders_every_n_epochs = 1 if self.training_params.data_streaming_train else 0,
                    limit_val_batches = self.training_params.limit_val_batches,
                    log_every_n_steps = self.training_params.log_every_n_steps,
                    # amp_backend="apex",
                    num_sanity_val_steps=2,
                    )
            
        else:

            if self.training_params.local_trainer == 'sentence':
                from micro_trainer_transformers.sentence_trainer import SentenceTrainer
                self.m_trainer = SentenceTrainer(**max_train_kwargs,
                        accumulate_grad_batches=self.training_params.gradient_accumulation_steps,
                        device='cuda',precision=precision,
                        gradient_clip_val=self.training_params.max_grad_norm,
                        val_check_interval=val_check_interval,
                        path_log=self.training_params.path_log,
                        path_checkpoints=self.training_params.path_checkpoint)
            else:

                from micro_trainer_transformers.core_trainer import LocalTrainer
                self.m_trainer = LocalTrainer(**max_train_kwargs,
                        accumulate_grad_batches=self.training_params.gradient_accumulation_steps,
                        device='cuda',precision=precision,
                        gradient_clip_val=self.training_params.max_grad_norm,
                        val_check_interval=val_check_interval,
                        path_log=self.training_params.path_log,
                        path_checkpoints=self.training_params.path_checkpoint)


    def model_vis_save(self):
        '''
        Сохраняет визуализацию сетки
        '''
        
        self.create_trainer()
        
        self.m_trainer.num_sanity_val_steps = 0
        self.m_trainer.limit_train_batches = 0
        self.m_trainer.limit_val_batches = 1
        
        self.save_first_valid_batch = True
        
        self.m_trainer.validate(self,ckpt_path=None)
        
        #https://github.com/mert-kurttutan/torchview
        from torchview import draw_graph
        import os.path

        print('=======================')
        print('Run save graph model:', self.training_params.path_log)
        
        old_size = 0
        for depth in range(100):

            save_name = 'model_'+self.training_params.model_master_name+'_depth_' + str(depth)

            _ = draw_graph(
                self.model, self.first_valid_batch,
                graph_name=save_name,
                depth=depth,
                save_graph=True,
                filename=self.training_params.path_log + save_name,
            )

            real_filename = self.training_params.path_log + save_name+'.png'

            new_size = os.path.getsize(real_filename)

            if new_size == old_size:
                break

            old_size = new_size
            
            print('Save graphviz: ',real_filename)  

    def lr_find(self,):
        '''
        Проводит поиск learning rate и возвращает оптимальное значение lr
        
        Кроме этого делает print
        и сохраняет график поиска learning rate в папку self.training_params.path_log
        
        '''
        
        
        self.create_trainer(mode = 'lr_find')
        
        self.train_mode_start = False
        
        lr_finder = self.m_trainer.tuner.lr_find(self, self.train_dataloader())
        
        #print(lr_finder.results)
        
        lr_recomend = lr_finder.suggestion()
        
        print('=======================')
        print('Recomended lr:', lr_recomend)
        
        fig = lr_finder.plot(suggest=True) # Plot
        fig.show()
        
        save_log_filename = self.training_params.path_log + 'find_lr.jpg'        
        fig.savefig(save_log_filename) 
        
        print('Save image plot result to:', save_log_filename)
        
        return lr_recomend
        
    def fit(self,resume_from_checkpoint=None, resume_sheduler: bool = True):
        
        self.create_trainer()

        save_useful_info(self.training_params.path_log,self.model,self.training_params.__dict__, self.m_trainer.__dict__)

        self.m_trainer.fit(self,ckpt_path=resume_from_checkpoint, resume_sheduler = resume_sheduler)
        
        if not self.training_params.local_trainer:
            print('Best model path:', self.checkpoint_callback.best_model_path)
            print(' --- model score:', self.checkpoint_callback.best_model_score)
        
