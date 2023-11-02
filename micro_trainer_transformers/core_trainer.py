from contextlib import nullcontext
from tqdm.auto import tqdm
import os
import pickle
import torch

from torch.utils.tensorboard import SummaryWriter

from .utils import make_dirs

class LocalDataModule:
    def __init__(self, mode, model):
        self.mode = mode
        self.model = model
        self.end_epoch = False

        self.dl_train = model.train_dataloader()
        self.dl_valid = model.val_dataloader()

        self.dl_train_iter = iter(self.dl_train)
        #self.dl_valid_iter = iter(dl_valid)

    def reset_epoch(self):
        self.end_epoch = False

    def get_train_batch(self):
        batch: None = next(self.dl_train_iter, None)

        if batch is None:
            if self.mode == 'epoch':
                self.end_epoch = True
            else:
                pass
            
            self.dl_train_iter = iter(self.dl_train)
            
            batch = next(self.dl_train_iter)

        return batch

class LocalTrainer:
    def __init__(self, max_epochs = None, max_steps = None,
                 accumulate_grad_batches=None, device=None,
                 precision=None,
                 gradient_clip_val=0.0,
                 val_check_interval=None,
                 path_log=None,
                 path_checkpoints=None):
        self.max_epochs = max_epochs
        self.max_steps = max_steps

        self.global_epoch = 0
        self.mode = 'epoch'
        if self.max_epochs is None:
            self.mode = 'steps'

        self.accumulate_grad_batches=accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.val_check_interval = val_check_interval

        self.global_step = 0


        self.device = device

        if precision is None:
            precision = 32

        dtype = 'float32'
        if precision == 'bf16':
            dtype = 'bfloat16'
        elif precision == 16 or precision == '16':
            dtype = 'float16'

        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

        self.dtype = dtype

        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)

        self.logger_tb = None
        if not path_log is None:
            self.logger_tb = SummaryWriter(os.path.join(path_log,'tensorboard'))

        self.path_checkpoints = path_checkpoints

        pass

    def batch_to_device(self, batch):
        if isinstance(batch,torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch,dict):
            for key in batch:
                batch[key] = batch[key].to(self.device)
        return batch


    def validate(self, pl_model):
        '''validate'''
        pl_model.eval()
        torch.set_grad_enabled(False)

        pl_model.on_validation_epoch_start()
        for batch_idx, batch in enumerate(tqdm(self.dm.dl_valid,desc='eval')):
            batch = self.batch_to_device(batch)
            pl_model.validation_step(batch,batch_idx=batch_idx)
        pl_model.on_validation_epoch_end()

        torch.set_grad_enabled(True)

    def log(self, metric_name, metric_value, **kwargs):
        '''
        log function
        '''
        #print(metric_name,metric_value)

        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()

        if not self.logger_tb is None:
            self.logger_tb.add_scalar(metric_name, metric_value, self.global_step)

    def log_dict(self, metric_dict, **kwargs):
        for key in metric_dict.keys():
            self.log(key, metric_dict[key], **kwargs)

    def save_checkpoint(self,pl_model,optimizers,lr_shedulers,current_step,current_epoch):

        if not self.path_checkpoints is None:
            folder_name = f'checkpoint_{current_step:08}'
            folder_name = os.path.join(self.path_checkpoints,folder_name)
            make_dirs(folder_name, silent=True)
            torch.save(pl_model.model.state_dict(), os.path.join(folder_name,'model.pt')) 
            chk_dict = {}
            chk_dict['optimizers'] = optimizers
            chk_dict['lr_shedulers'] = lr_shedulers
            chk_dict['current_step'] = current_step
            chk_dict['current_epoch'] = current_epoch
            pickle.dump(chk_dict, open(os.path.join(folder_name,'options.pkl'), 'wb'))

    def load_checkpoint(self,pl_model,folder_name):
            pl_model.model.load_state_dict(torch.load(os.path.join(folder_name,'model.pt')))
            return pickle.load(open(os.path.join(folder_name,'options.pkl'), 'rb'))

    def fit(self, pl_model=None, ckpt_path=None):
        #replace function
        pl_model.log = self.log
        pl_model.log_dict = self.log_dict

        pl_model.to(self.device)
        # initialize a GradScaler. If enabled=False scaler is a no-op
        #scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        pl_model.prepare_data()

        pl_model.on_fit_start()

        pl_model.setup("fit")

        self.dm = LocalDataModule(self.mode, pl_model)

        optimizers, lr_shedulers = pl_model.configure_optimizers()

        train_max_step = 0
        if self.mode == 'steps':
            train_max_step = self.max_steps
        else:
            max_steps = self.max_epochs * len(self.dm.dl_train)
            max_steps = max_steps // self.accumulate_grad_batches
            train_max_step=max_steps

        progress_bar = tqdm(total=train_max_step,desc='Train')
        
        pl_model.train()

        pl_model.on_train_start()

        #fit loop 
        #https://pytorch-lightning.readthedocs.io/en/1.8.6/common/lightning_module.html

        iter_num = 0
        epoch_num = 0

        if not ckpt_path is None:
            print('Resume from checkpoint:',ckpt_path)
            chk_dict = self.load_checkpoint(pl_model,ckpt_path)
            optimizers = chk_dict['optimizers']
            lr_shedulers = chk_dict['lr_shedulers']
            iter_num = chk_dict['current_step']
            epoch_num = chk_dict['current_epoch']
            for _ in range(iter_num):
                _ = self.dm.get_train_batch()
                progress_bar.update(1)
            
            #validate
            self.validate(pl_model)
            pl_model.train()

        pl_model.on_train_epoch_start()

        while True:
            self.global_step = iter_num

            #log lr
            if len(optimizers) > 0:
                current_lr = optimizers[0].param_groups[0]['lr']
                self.log('learning_rate', current_lr)

            for micro_step in range(self.accumulate_grad_batches):

                batch_idx = 0

                batch = self.dm.get_train_batch()

                # for key in batch.keys():
                #     batch[key] = batch[key].half()

                if micro_step == 0:
                    pl_model.on_train_batch_start(batch, batch_idx)

                batch = self.batch_to_device(batch)

                loss = pl_model.training_step(batch,batch_idx=batch_idx)
                #scaler.scale(loss).backward()
                loss.backward()
            
            # clip the gradient
            if self.gradient_clip_val != 0.0:
                # for optimizer in optimizers:
                #     scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pl_model.parameters(), self.gradient_clip_val)

            # step the optimizers and scaler if training in fp16
            for optimizer in optimizers:
                #scaler.step(optimizer)
                optimizer.step()

            #scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            #model.on_train_batch_end()

            #Проверка нужно ли запустить валидацинный цикл
            validate_epoch_end = False
            if not self.val_check_interval is None:
                if iter_num % self.val_check_interval == 0 and iter_num != 0:
                    validate_epoch_end = True
            elif self.dm.end_epoch:
                validate_epoch_end = True
            else:
                pass

            iter_num += 1

            if validate_epoch_end:
                self.save_checkpoint(pl_model,optimizers,lr_shedulers,iter_num,epoch_num)
                self.validate(pl_model)
                pl_model.train()
                pl_model.on_train_epoch_start()

                if self.mode == 'epoch':
                    self.dm.reset_epoch()
                    epoch_num += 1

                    for sheduler in lr_shedulers:
                        if sheduler['interval'] == 'epoch':
                            sheduler['scheduler'].step()

                    if self.max_epochs >= epoch_num:
                        break

            #exit from training loop
            if iter_num > train_max_step:
                break

            for sheduler in lr_shedulers:
                if sheduler['interval'] == 'step':
                    sheduler['scheduler'].step()

            progress_bar.update(1)

        pl_model.on_train_end()

        progress_bar.close()

        pl_model.on_fit_end()
