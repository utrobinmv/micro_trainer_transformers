import torch


class TrainerLogger:

    store_log_dict: dict = {}
    store_global_step = -1

    def log(self, metric_name, metric_value, **kwargs):
        '''
        log function
        '''
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()

        if self.store_global_step != self.global_step:
            for key in self.store_log_dict.keys():
                values = self.store_log_dict[key]
                if len(values) > 0:
                    value = sum(values) / len(values)

                    if self.logger_tb:
                        self.logger_tb.add_scalar(key, value, self.store_global_step)
            self.store_log_dict = {}

        if metric_name not in self.store_log_dict.keys():
            self.store_log_dict[metric_name] = []
        self.store_log_dict[metric_name].append(metric_value)

        self.store_global_step = self.global_step

    def log_dict(self, metric_dict, **kwargs):
        for key in metric_dict.keys():
            self.log(key, metric_dict[key], **kwargs)
