import torch
import pandas as pd
from .core_trainer import LocalTrainer
from .utils import in_jupyter_notebook


class SentenceTrainer(LocalTrainer):
    def validate(self, pl_model):
        '''validate'''
        if pl_model.evaluator:
            pl_model.eval()
            torch.set_grad_enabled(False)

            eval_dict = pl_model.model.evaluate(evaluator=pl_model.evaluator)
            self.log_dict(eval_dict)

            if len(eval_dict.keys()) > 0:
                dict_result = {}
                dict_result['step'] = self.global_step
                for key in eval_dict.keys():
                    dict_result[key] = eval_dict[key]
                pl_model.epoch_valid_list_dict_result.append(dict_result)
                if in_jupyter_notebook():
                    from IPython.display import display, HTML
                    pd.set_option('display.max_rows', None)
                    df = pd.DataFrame(pl_model.epoch_valid_list_dict_result)
                    display(df)
                    del df
                else:
                    print(pl_model.epoch_valid_list_dict_result[-1])

            torch.set_grad_enabled(True)
