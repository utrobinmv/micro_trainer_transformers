from typing import Any, Callable, Dict, NewType, Union, List, Optional, Tuple

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import datasets
import pytorch_lightning as pl

from micro_trainer_transformers import TrainigParameters
from .utils import time_in_second_hms, make_dirs, set_seed, save_useful_info
from .core_optim import UniversalOptim

class UniversalTrainingModule(pl.LightningModule, UniversalOptim):
    def __init__(
        self,
        model: nn.Module,
        args: TrainigParameters, 
        train_dataset: datasets.Dataset,
        eval_dataset: datasets.Dataset,
        data_collator: Any = None,
        compute_metrics: Any = None,
        
    ) -> None:
        super().__init__()

        self.trainer = None

        self.training_params = args
        
        self.dataset_train = train_dataset
        self.dataset_valid = eval_dataset
        
        if self.training_params.data_streaming_train:
            self.dataset_train_size = 0
            self.dataset_train_size_count = 0 #Подсчет размера итеративного датасета
            self.dataset_train_iter = iter(self.dataset_train)
        else:
            self.dataset_train_size = len(self.dataset_train)

        if self.training_params.data_streaming_valid:
            self.dataset_valid_size = 0
            self.dataset_valid_size_count = 0 #Подсчет размера итеративного датасета
            self.dataset_valid_iter = iter(self.dataset_valid)
        else:
            self.dataset_valid_size = len(self.dataset_valid)
        self.dataloader_valid = None
        self.dataset_total_batches = 0
        
        #for resume checkpoint
        self.dataset_start_epoch_data_iter = 0
        self.pbar_train_restore_n = 0
        self.checkpoint_callback = None
        
        self.model = model
        self.data_collator = data_collator
        self.compute_metrics_fn = compute_metrics
        
        self.loss_fn = self.training_params.loss_fn
        
        self.pbar_train = tqdm(desc='Total training model',leave=True,position=2) 
        
        self.train_mode_start = False 

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        pass

    def on_fit_start(self):
        #print('on fit start')
        #self.current_training_step = iter(range(self.training_params.max_train_steps))
        #self.pbar_train = iter(tqdm(range(self.training_params.max_train_steps),desc='Training model',leave=True))
        
        self.train_mode_start = True
        
        self.pbar_train.clear()
        self.pbar_train.reset(total=self.training_params.max_train_steps)
        if self.pbar_train_restore_n > 0: #restore from checkpoint
            self.pbar_train.n = self.pbar_train_restore_n
            self.pbar_train_restore_n = 0
        
        pass 

    def on_train_start(self):
        self.validate_labels = []
        self.validate_predictions = []
        
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
        return super().configure_optimizers_optim()

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
        checkpoint['pbar_train_n'] = self.pbar_train.n
        checkpoint['dataset_train_size'] = self.dataset_train_size

        pass

        
    def on_train_batch_start(self, batch, batch_idx):
        
        if batch_idx % self.training_params.gradient_accumulation_steps == 0 and self.train_mode_start:
            # if self.pbar_train.total == self.pbar_train.n:
            #     self.pbar_train.total += 1
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
                data_epoch = self.global_step / self.dataset_total_batches
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
               
        if self.training_params.loss_fn_in_model:
            preds = self.model(**batch)
            loss = preds.loss
            
            if valid:
                if self.compute_metrics_fn:
                    self.validate_labels.extend(list(batch['labels'].detach().cpu()))
                    self.validate_predictions.extend(list(preds.logits.detach().cpu()))
            
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
        
        self.log("training_loss", loss, on_step=True, 
                 on_epoch=True, prog_bar=False, 
                 batch_size=batch_size)
             
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], batch_idx: int,
    ) -> torch.Tensor:
        
        if isinstance(batch, dict):
            batch_size = len(batch[list(batch.keys())[0]])
        else:
            batch_size = len(batch["labels"])
            
        loss = self.common_step(batch, batch_idx, valid = True)
        
        self.log("valid_loss", loss, on_step=False, 
                 on_epoch=True, prog_bar=True, 
                 batch_size=batch_size)

    def on_validation_epoch_start(self):
        self.validate_labels = []
        self.validate_predictions = []
        
        pass

    def on_validation_epoch_end(self):
        if self.compute_metrics_fn:            
            
            result_dict = self.compute_metrics_fn(
                (pad_sequence(self.validate_predictions, batch_first=True, padding_value=-100), 
                 pad_sequence(self.validate_labels, batch_first=True, padding_value=-100))
                )
            #result_dict = self.compute_metrics_fn((torch.vstack(self.validate_predictions), torch.vstack(self.validate_labels)))

            if len(result_dict.keys()) > 0:
                self.log_dict(result_dict, on_step=False, on_epoch=True, prog_bar=False)
            
        pass


    def create_trainer(self, mode='train'):

        set_seed(self.training_params.seed)
        
        if self.training_params.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        
        loggers = []
        
        make_dirs(self.training_params.path_log + 'wandb/')
        
        loggers.append(pl.loggers.TensorBoardLogger(save_dir = self.training_params.path_log + 'tensorboard',
                                            name = None,
                                            version = '',
                                            ))
        loggers.append(pl.loggers.CSVLogger(save_dir = self.training_params.path_log + 'csv_log',
                                            name = None,
                                            version = '',
                                            ))
        
        if not self.training_params.debug and mode == 'train':
            
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
            val_check_interval = (self.training_params.val_check_interval-1)*self.training_params.gradient_accumulation_steps
            check_val_every_n_epoch=None
        elif self.training_params.evaluation_strategy == 'epoch':
            val_check_interval = None
            check_val_every_n_epoch=1
        
        precision = 32
        if self.training_params.bf16:
            precision = "bf16"
        elif self.training_params.fp16:
            precision = 16
        
        self.trainer = pl.Trainer(accelerator="auto", devices="auto", 
                max_epochs=self.training_params.max_train_epochs, max_steps=self.training_params.max_train_steps,
                val_check_interval=val_check_interval,
                accumulate_grad_batches=self.training_params.gradient_accumulation_steps,
                check_val_every_n_epoch=check_val_every_n_epoch,
                default_root_dir=self.training_params.path_best_model,
                logger=loggers,
                gradient_clip_val=self.training_params.max_grad_norm,
                callbacks=callbacks,
                reload_dataloaders_every_n_epochs = 1 if self.training_params.data_streaming_train else 0,
                limit_val_batches = self.training_params.limit_val_batches,
                log_every_n_steps = self.training_params.log_every_n_steps,
                num_sanity_val_steps=2,
                )


    def lr_find(self,):
        self.create_trainer(mode = 'lr_find')
        
        self.train_mode_start = False
        
        lr_finder = self.trainer.tuner.lr_find(self, self.train_dataloader())
        
        #print(lr_finder.results)
        
        print('=======================')
        print('Recomended lr:', lr_finder.suggestion())
        
        fig = lr_finder.plot(suggest=True) # Plot
        fig.show()
        
        save_log_filename = self.training_params.path_log + 'find_lr.jpg'        
        fig.savefig(save_log_filename) 
        
        print('Save image plot result to:', save_log_filename)
        
    def fit(self,resume_from_checkpoint=None):
        
        self.create_trainer()

        save_useful_info(self.training_params.path_log,self.model,self.training_params.__dict__, self.trainer.__dict__)

        self.trainer.fit(self,ckpt_path=resume_from_checkpoint)
        
        print('Best model path:', self.checkpoint_callback.best_model_path)
        print(' --- model score:', self.checkpoint_callback.best_model_score)
        
