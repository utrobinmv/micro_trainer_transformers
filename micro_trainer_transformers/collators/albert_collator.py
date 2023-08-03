from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import DataCollatorForWholeWordMask
import random

@dataclass
class LM_DataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    '''
    Замена на mask токены целых слов, вместо отдельных токенов
    
    data_collator = LM_DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.1
    ) 
    '''
    def set_param_diffusion(self, value_mask_token = 0.8, replace_tokens = 0.5):
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        #Какой процент оставить без изменения на шум
        #1.0 - шума нет (второй параметр ничего не меняет)
        #0.8 - Шума 20% процентов (по умолчанию)
        #0.5 - Шума 50%
        #0.0 - Всё маскированные токены, заменяются на шум
        self.value_mask_token = value_mask_token

        # 10% of the time, we replace masked input tokens with random word
        #Сколько из шумных токенов будут заменены на случайные токены, или вернуты на не маск
        #1.0 - все зашумленные токены будут заменены на случайные
        #0.5 - половина токенов маск будет заменена на случайные токены (по умолчанию)
        #0.1 - практически все замаскированные токены останутся замаскированными
        #0.0 - все замаскированные токены останутся замаскированными
        
        self.replace_tokens = replace_tokens

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        input_tokens = построчно, токенизированный текст. Не батчем, а для каждого элемента
        input_tokens == ['[CLS]', '▁Э', 'т', 'а', '▁х', 'а', 'р', 'а', 'к', 'т', 'е', 'р', 'и', 'с', ...]
        """
        # if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
        #     warnings.warn(
        #         "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
        #         "Please refer to the documentation for more information."
        #     )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token != "▁" and len(token) == 1:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        
        inputs - входной батч. torch.Size([batch_size, 150])
        
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        #probability_matrix.masked_fill_(~special_tokens_mask, value=0.0)
        #masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = special_tokens_mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.value_mask_token)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        #print('Ok')

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, self.replace_tokens)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
