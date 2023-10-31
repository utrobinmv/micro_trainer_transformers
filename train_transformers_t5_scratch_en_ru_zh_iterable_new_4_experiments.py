#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install langdetect
#!pip install transformers==4.31.0


# In[ ]:


import os
import time
import pickle
import numpy as np
import torch
import datasets
from tqdm.auto import tqdm
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# In[ ]:


from transformers import AutoConfig, T5Tokenizer, T5TokenizerFast, AutoModel, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


# In[ ]:


import math
import statistics
import kenlm
import nltk
from nltk.tokenize import sent_tokenize
import evaluate
from langdetect import detect


# In[ ]:


#!pip install langdetect


# In[ ]:


#root_dir = '/'


# In[ ]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('punkt')


# In[ ]:


debug = False

etap_train = 6

if etap_train == 6:
    #batch_size = 32
    batch_size = 16
    gradient_accumulation_steps = 2
    max_input_length = 200
    max_target_length = 200

    train_steps = 19548 #Устанавливается на втором шаге
    num_train_epochs = 1.1
    eval_steps = 1000
    save_steps = 999

    eval_steps = 5
    save_steps = 5

    warmup_steps = 2000
    save_total_limit = 10
    learning_rate=2e-2
    learning_rate_final_cosine=1e-5

num_workers = 4

optim_weight_decay = 0.0

print('etap =',etap_train,learning_rate)

streaming = True


# In[ ]:


rouge_score = evaluate.load("rouge")
sacrebleu_score = evaluate.load("sacrebleu")
#meteor_score = evaluate.load('meteor')
bertscore_ru = evaluate.load("bertscore")
bertscore_en = evaluate.load("bertscore")
bertscore_zh = evaluate.load("bertscore")
metric_chrf = evaluate.load("chrf")


# In[ ]:


os.listdir('/data')


# In[ ]:


pp_model_ru = kenlm.Model(f'/data/kenlm/' + f'ru.arpa.bin')
pp_model_en = kenlm.Model(f'/data/kenlm/' + f'en.arpa.bin')
pp_model_zh = kenlm.Model(f'/data/kenlm/' + f'zh.arpa.bin')


# In[ ]:




# In[ ]:


from data.read_data_fast import FixLenFastReadDS


# In[ ]:


data_dir = '/share/datasets_bin/en_ru_zh_translate_corpus_freq_2'


# In[ ]:


tokenizer = T5TokenizerFast.from_pretrained('/data/tokenizers/tokenizer_t5_en_ru_zh_65000', max_len=max_input_length)
tokenizer_slow= T5Tokenizer.from_pretrained('/data/tokenizers/tokenizer_t5_en_ru_zh_65000', max_len=max_input_length)
if etap_train == 6:
    model_name = '/data/models/t5_scratch_translate_en_ru_zh_2023_10_16_20-36-18'
    #model = T5ForConditionalGeneration.from_pretrained(model_name,torch_dtype=torch.float16)
    model = T5ForConditionalGeneration.from_pretrained(model_name)


# In[ ]:


model.dtype


# In[ ]:


# if etap_train == 4:
#     #Заморозим все слои кроме эмбедингов
#     for param in model.parameters():
#         param.requires_grad = False 
#     for param in model.shared.parameters():
#         param.requires_grad = True 
#     for param in model.decoder.embed_tokens.parameters():
#         param.requires_grad = True 
#     for param in model.encoder.embed_tokens.parameters():
#         param.requires_grad = True 


# In[ ]:


for param in model.parameters():
    print(param.requires_grad,end=', ')


# In[ ]:

def replace__len__(a):
    return 32*30

FixLenFastReadDS.__len__ = replace__len__


rd_val = FixLenFastReadDS(data_dir,'val',max_input_length,tokenizer_slow)
#rd_train = FixLenReadDS(data_dir,'val',max_input_length,tokenizer)
rd_train = FixLenFastReadDS(data_dir,'val',max_input_length,tokenizer_slow)


# In[ ]:


len(rd_val), len(rd_train)


# In[ ]:


#rd_val[6]


# In[ ]:


np_input, np_label = rd_val[0]['input_ids'], rd_val[0]['labels']
tokenizer.decode(np_input), tokenizer.decode(np_label)


# In[ ]:


#Заморозим все слои кроме эмбедингов
# for param in model.encoder.parameters():
#     param.requires_grad = False 
# for param in model.shared.parameters():
#     param.requires_grad = False 
# for param in model.decoder.embed_tokens.parameters():
#     param.requires_grad = False 


# In[ ]:


def current_time_in_second(): 
    #Возвращает время в секундах
    return round(time.time())


# In[ ]:


def time_in_second_to_textdate(time_in_second):  
    #Преобразовывает секунды в текстовую дату
    local_time = time.localtime(time_in_second)
    str_time = time.strftime("%Y_%m_%d_%H-%M-%S", local_time)
    return str_time


# In[ ]:


model_save_name = f't5_scratch_translate_en_ru_zh_'+time_in_second_to_textdate(current_time_in_second())
model_save_name


# In[ ]:


args = Seq2SeqTrainingArguments(
   f"/checkpoints/{model_save_name}/ckpts",
   evaluation_strategy = "steps",
   learning_rate=learning_rate,
   per_device_train_batch_size=batch_size,
   per_device_eval_batch_size=batch_size,
   report_to='tensorboard',
   logging_dir=f'/checkpoints/{model_save_name}/logs',
   #weight_decay=0.01,
   save_total_limit=save_total_limit,
   #lr_scheduler_type = 'cosine', #['linear', 'cosine', 
   #'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt']
   #lr_scheduler_type = 'constant_with_warmup',

   metric_for_best_model='Chrf',
   greater_is_better=True,
   remove_unused_columns=False,
   load_best_model_at_end=False,
    
   dataloader_num_workers=num_workers,

   #warmup_steps=train_steps//20,

   num_train_epochs=num_train_epochs,
   #max_steps=train_steps,
   eval_steps=eval_steps,
   save_steps=save_steps,
    
   logging_steps=min(50,eval_steps,save_steps),
   gradient_accumulation_steps=gradient_accumulation_steps,
   save_strategy="steps",
   tf32=True,
   bf16=False,
   fp16=True,
   torch_compile=True,
   #optim="adamw_torch_fused",
   adafactor=True,
    
   predict_with_generate=True,
   generation_max_length=max_target_length,
   #generation_config = model.generation_config,
   #ignore_data_skip=True,
)


# In[ ]:


from torch.optim import Optimizer
from typing import Iterable, Tuple
from torch import nn

from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )


# In[ ]:


class AdamWScale(Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # /Adapt Step from Adafactor
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adafactor

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


# In[ ]:


no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": optim_weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamWScale(
    optimizer_grouped_parameters,
    lr=learning_rate,
)


# In[ ]:


scheduler1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=warmup_steps,
        last_epoch=-1,
    )
scheduler2 = CosineAnnealingLR(
    optimizer,
    T_max=train_steps - warmup_steps,
    eta_min=learning_rate_final_cosine,
)
lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps]
    )


# In[ ]:


class CollatorTransformers:
    def __init__(self, pad_value=0, mask_pad_value=0, label_pad_value=-100):
        self.pad_value = pad_value
        self.mask_pad_value = mask_pad_value
        self.label_pad_value = label_pad_value

    def __call__(self, batch):
        '''
        Специфика данного коллатора что он заточен именно под сетки marian
        особенности в том, что pad_token = 1
        а в метках паддинг заменяется последовательность -100
        '''
        #np_input, np_label, len_input_ids, len_labels = batch

        #print(batch)
        
        #dict_batch = {k: [v] for k, v in batch.items()}
        dict_batch = {k: [dic[k] for dic in batch] for k in batch[0].keys()}

        batch_size = len(dict_batch['input_ids'])

        # print(batch.keys())
        # print(batch_size)
        # print(batch)
        dict_batch['input_ids'] = [torch.tensor(x.astype(np.int32)) for x in dict_batch['input_ids']]
        dict_batch['labels'] = [torch.tensor(x.astype(np.int64)) for x in dict_batch['labels']]        
        
        dict_batch['attention_mask'] = [torch.ones_like(x) for x in dict_batch['input_ids']]
        #print(batch['attention_mask'])

        dict_batch.pop('len_input_ids')
        dict_batch.pop('len_labels')

        #batch['attention_mask'] = [torch.tensor(x) for x in batch['attention_mask']]
        
        dict_batch['input_ids'] = pad_sequence(dict_batch['input_ids'], batch_first=True, padding_value=self.pad_value)
        dict_batch['attention_mask'] = pad_sequence(dict_batch['attention_mask'], batch_first=True, padding_value=self.mask_pad_value)
        dict_batch['labels'] = pad_sequence(dict_batch['labels'], batch_first=True, padding_value=self.label_pad_value)

        # print(batch.keys())
        # print(batch['input_ids'].shape)
        # print(batch['attention_mask'].shape)
        # print(batch['labels'].shape)
        # print(batch['input_ids'].dtype)
        # print(batch['attention_mask'].dtype)
        # print(batch['labels'].dtype)
        
        return dict_batch


# In[ ]:


data_collator = CollatorTransformers(tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id)
#data_collator = CollatorTransformers(tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id)
tokenizer.pad_token_id, tokenizer.eos_token_id


# In[ ]:


dl = DataLoader(rd_val,batch_size=batch_size,collate_fn=data_collator)


# In[ ]:


it = iter(dl)


# In[ ]:


next(it).keys()


# In[ ]:


def get_perplexity(pp_model, s):
    if isinstance(s, str):
        n = len(s.split())
        sum_inv_logs = -1 * sum(score for score, _, _ in pp_model.full_scores(s))
        if n == 0:
            return 0
        return math.pow(sum_inv_logs, 1.0/n)
    return 0.0


# In[ ]:


def compute_metrics(references, predictions):
    '''
    references, predictions
    '''
    #print('calculate rouge')
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in predictions]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in references]
    
    # print('predictions:',predictions[0])
    # print('references:',references[0])

    # print('decoded_preds:',decoded_preds[0])
    # print('decoded_labels:',decoded_labels[0])
    # print('-----------------')
    
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value * 100 for key, value in result.items()}

    #print('calculate bleu')
    sacrebleu_result = sacrebleu_score.compute(predictions=predictions, references=references)
    
    #print('calculate chrf')
    chrf_result = metric_chrf.compute(predictions=predictions, references=references)
    #result["chrf"] = chrf_result["score"]
    result["eval_Chrf"] = chrf_result["score"]

    list_ru_references = []
    list_en_references = []
    list_zh_references = []
    list_ru_predictions = []
    list_en_predictions = []
    list_zh_predictions = []
    for label, predict in zip(references,predictions):
        lang = detect(label)
        if  lang == 'ru':
            list_ru_references.append(label)
            list_ru_predictions.append(predict)
        elif lang == 'en':
            list_en_references.append(label)
            list_en_predictions.append(predict)
        else:
            list_zh_references.append(label)
            list_zh_predictions.append(predict)
        
    
    #meteor_result = meteor_score.compute(predictions=predictions, references=references)
    
    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    #result["gen_len"] = np.mean(prediction_lens)

    result['sacrebleu'] = sacrebleu_result['score']
    #result["meteor"] = meteor_result["meteor"]
    
    #print('calculate ppl')
    list_ppl_ru = []
    list_ppl = []
    for predict in list_ru_predictions:
        ppl = get_perplexity(pp_model_ru, predict)
        list_ppl_ru.append(ppl)
        list_ppl.append(ppl)
    result["ppl_ru"] = statistics.mean(list_ppl_ru)

    list_ppl_en = []
    for predict in list_en_predictions:
        ppl = get_perplexity(pp_model_en, predict)
        list_ppl_en.append(ppl)
        list_ppl.append(ppl)
    result["ppl_en"] = statistics.mean(list_ppl_en)

    list_ppl_zh = []
    for predict in list_zh_predictions:
        ppl = get_perplexity(pp_model_zh, predict)
        list_ppl_zh.append(ppl)
        list_ppl.append(ppl)
    result["ppl_zh"] = statistics.mean(list_ppl_zh)
    result["ppl"] = statistics.mean(list_ppl)

    #print('calculate bertscore ru')
    #results_bert_ru = bertscore_ru.compute(predictions=list_ru_predictions, references=list_ru_references, lang='ru', device='cpu')
    #print('calculate bertscore en')
    #results_bert_en = bertscore_en.compute(predictions=list_en_predictions, references=list_en_references, lang='en', device='cpu')
    #print('calculate bertscore zh')
    #results_bert_zh = bertscore_zh.compute(predictions=list_zh_predictions, references=list_zh_references, lang='zh', device='cpu')
    # print(results_bert_ru['hashcode'])
    # print(results_bert_en['hashcode'])
    # print(results_bert_zh['hashcode'])
    # for key in ['precision','recall','f1']:
    #     result['bertscore_ru_' + key] = statistics.mean(results_bert_ru[key])
    #     result['bertscore_en_' + key] = statistics.mean(results_bert_en[key])
    #     result['bertscore_zh_' + key] = statistics.mean(results_bert_zh[key])

    return {k: round(v, 4) for k, v in result.items()}    


# In[ ]:


def compute_metrics_decode(eval_pred):
    predictions, labels = eval_pred
    #print('=======================================')
    # Decode generated summaries into text
    
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    #print('predictions',predictions)
    #print('labels',labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return compute_metrics(decoded_labels, decoded_preds)


# In[ ]:




trainer = Seq2SeqTrainer(
   model,
   args,
   train_dataset=rd_train,
   eval_dataset=rd_val,
   data_collator=data_collator,
   compute_metrics=compute_metrics_decode,
   optimizers=(optimizer,lr_scheduler),
)


# In[ ]:


#torch.cuda.amp.autocast(True)


# In[ ]:


#model- float32 train - bfloat16 - start 14700 and out of memory
#model- float32 train - float16 - start 14700 and out of memory 46:23


# In[ ]:


#!pip install micro_trainer_transformers


# In[ ]:


from micro_trainer_transformers import TrainigParameters, UniversalTrainingModule


# In[ ]:


args.save_steps = args.eval_steps

#trainer.train()

#stop

# In[ ]:


training_param = TrainigParameters()
training_param.debug = False
training_param.from_hf_training_arguments('t5_exp', 't5_exp_1',args)


# In[ ]:


#!pip freeze


# In[ ]:


trainer = UniversalTrainingModule(
    model=model,
    args=training_param,
    data_collator=data_collator,
    train_dataset=rd_train,
    eval_dataset=rd_val,
    optimizers=(optimizer,lr_scheduler),
    compute_metrics=compute_metrics_decode    
    )


# In[ ]:


trainer.fit(resume_from_checkpoint=None)


# In[ ]:


stop


# In[ ]:




# In[ ]:


#stop


# In[ ]:


#trainer.train(resume_from_checkpoint='/checkpoints/t5_base_scratch_translate_en_ru_zh_2023_10_15_08-36-08/ckpts/checkpoint-999')


# In[ ]:


# trainer.compute_metrics=compute_metrics_decode
# eval_results = trainer.evaluate()
# eval_results


# In[ ]:


# model_name = f'{root_dir}data/checkpoints/models/marian_en_ru_first4-finetuned/checkpoint-90000'
# model_name


# In[ ]:


model.save_pretrained('/data/models/' + model_save_name)
tokenizer.save_pretrained('/data/models/' + model_save_name)
#torch.save(model.state_dict(), '/data/models/' + model_save_name + "/model.pt") 
model_save_name


# In[ ]:


#model = T5ForConditionalGeneration.from_pretrained('/data/models/' + model_save_name)


# In[ ]:


eval_iter = iter(rd_train)


# In[ ]:


#eval_iter = iter(ds_validation)


# In[ ]:


data = next(eval_iter)
tokenizer.batch_decode(torch.tensor([data['input_ids'].astype(np.int32)]), skip_special_tokens=True)


# In[ ]:


#data


# In[ ]:


tokenizer.batch_decode(torch.tensor([data['input_ids'].astype(np.int32)]), skip_special_tokens=False)


# In[ ]:


model.device


# In[ ]:


out = model.generate(torch.tensor([data['input_ids'].astype(np.int32)]).to(model.device),max_length=max_target_length)


# In[ ]:


tokenizer.batch_decode([data['labels'].astype(np.int32)], skip_special_tokens=True)


# In[ ]:


decoded_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
decoded_preds


# In[ ]:


text = 'translate to en: Привет как дела? У меня всё хорошо. А у тебя как дела? Я из Австрии. Давай пойдем вместе на улцу?'
text


# In[ ]:


b = tokenizer(text)
b


# In[ ]:


out = model.generate(torch.tensor([b['input_ids']]).to(model.device),max_length=max_target_length)
text2 = tokenizer.batch_decode(out, skip_special_tokens=True)
text2


# In[ ]:


b = tokenizer('translate to ru: '+text2[0])
b


# In[ ]:


out = model.generate(torch.tensor([b['input_ids']]).to(model.device),max_length=max_target_length)
text3 = tokenizer.batch_decode(out, skip_special_tokens=True)
text3


# In[ ]:





# In[ ]:


list_splitters = {'. ':'',' | ':'',' :: ':''}
list_splitters


# In[ ]:


list(list_splitters)


# In[ ]:




