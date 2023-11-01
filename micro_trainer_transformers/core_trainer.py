from contextlib import nullcontext
from tqdm.auto import tqdm
import os
import torch

from torch.utils.tensorboard import SummaryWriter

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
                 path_log=None):
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

        pass

    def batch_to_device(self, batch):
        if isinstance(batch,torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch,dict):
            for key in batch:
                batch[key] = batch[key].to(self.device)
        return batch


    def validate(self, model):
        '''validate'''
        model.eval()
        torch.set_grad_enabled(False)

        model.on_validation_epoch_start()

        for batch_idx, batch in enumerate(tqdm(self.dm.dl_valid,desc='eval')):
            batch = self.batch_to_device(batch)
            model.validation_step(batch,batch_idx=batch_idx)

        model.on_validation_epoch_end()

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

    def fit(self, model=None, ckpt_path=None):
        print("FIT FIT FIT")
        print(model)
        print("FIT FIT FIT")

        #replace function
        model.log = self.log
        model.log_dict = self.log_dict

        model.to(self.device)
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        model.prepare_data()

        model.on_fit_start()

        model.setup("fit")

        self.dm = LocalDataModule(self.mode, model)

        optimizers, lr_shedulers = model.configure_optimizers()

        train_max_step = 0
        if self.mode == 'steps':
            train_max_step = self.max_steps
        else:
            max_steps = self.max_epochs * len(self.dm.dl_train)
            max_steps = max_steps // self.accumulate_grad_batches
            train_max_step=max_steps

        progress_bar = tqdm(total=train_max_step,desc='Train')
        
        model.train()

        model.on_train_start()

        #fit loop 
        #https://pytorch-lightning.readthedocs.io/en/1.8.6/common/lightning_module.html

        model.on_train_epoch_start()

        iter_num = 0
        epoch_num = 0
        while True:
            self.global_step = iter_num

            #log lr
            if len(optimizers) > 0:
                current_lr = optimizers[0].param_groups[0]['lr']
                self.log('learning_rate', current_lr)

            for micro_step in range(self.accumulate_grad_batches):

                batch_idx = 0

                batch = self.dm.get_train_batch()

                if micro_step == 0:
                    model.on_train_batch_start(batch, batch_idx)

                batch = self.batch_to_device(batch)

                loss = model.training_step(batch,batch_idx=batch_idx)
                scaler.scale(loss).backward()
            
            # clip the gradient
            if self.gradient_clip_val != 0.0:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

            # step the optimizers and scaler if training in fp16
            for optimizer in optimizers:
                scaler.step(optimizer)

            scaler.update()
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

            if validate_epoch_end:
                self.validate(model)
                model.train()

                if self.mode == 'epoch':
                    self.dm.reset_epoch()
                    epoch_num += 1

                    for sheduler in lr_shedulers:
                        if sheduler['interval'] == 'epoch':
                            sheduler['scheduler'].step()

                    if self.max_epochs >= epoch_num:
                        break

            iter_num += 1

            #exit from training loop
            if iter_num > train_max_step:
                break

            for sheduler in lr_shedulers:
                if sheduler['interval'] == 'step':
                    sheduler['scheduler'].step()

            progress_bar.update(1)

        model.on_train_end()

        progress_bar.close()

        model.on_fit_end()

        print("FIT FIT FIT END")
