import os
import random
import pickle
import numpy as np
from tqdm.auto import tqdm

from .read_data import ReadDS

class FixLenFastReadDS(ReadDS):
    '''
    Класс получения данных, и удлинения их до max_len
    Можно получить очень сложных датасет для перевода с 
    длинами последовательностей 4к
    '''
    def __init__(self, data_dir, part, max_len=100, tokenizer=None):
        super().__init__(data_dir,part,max_len-2) #-2 чтобы без коллизий по длине сплиттера
        
        #prepare the separators
        list_splitters = {'. ':'',' | ':'',' :: ':''}
        list_splitters_keys = list(list_splitters.keys())
        for key in list_splitters_keys:
            s = tokenizer(key)[:-1]
            list_splitters[key] = s
        self.list_splitters = list_splitters

        local_index = np.arange(self.np_index.shape[0])
        list_langs = list(set(self.np_index[:,1]))
        dict_langs_len_prefix = {1:5,2:4,3:5}

        #Сформируем словарь индексов
        dict_langs_tmp = {}
        dict_parts = {}
        for lang in list_langs:
            dict_langs_tmp[lang] = local_index[self.np_index[:,1] == lang]
            dict_parts[lang] = list(self.np_index[self.np_index[:,1] == lang,0])

        dict_langs = {}
        for lang in list_langs:
            dict_langs[lang] = {}
            parts = set(list(dict_parts[lang]))
            for part in parts:
                indices = [index for index, value in enumerate(dict_parts[lang]) if value == part]
                dict_langs[lang][part] = list(dict_langs_tmp[lang][indices])


        step_view = 0
        self.filename_len_index = f'{self.data_dir}/{self.part}_fast_len_index_{max_len}.pkl'
        if os.path.exists(self.filename_len_index):
            self.fix_sentence = pickle.load(open(self.filename_len_index, 'rb'))
        else:
            self.fix_sentence = []
            #self.fix_sentence_len = []
            while True:

                if step_view % 100 == 0:
                    # создаем список вероятностей выбора ключей
                    #prob_list = [len(dict_langs_tmp[key]) for key in dict_langs]
                    prob_list = []
                    for lang in dict_langs.keys():
                        l = [len(dict_langs[lang][key]) for key in dict_langs[lang]]
                        prob_list.append(sum(l))

                    # Проверим на исчерпание списка
                    if sum(prob_list) == 0:
                        break

                    # нормализуем список вероятностей
                    random_prob_list = [prob/sum(prob_list) for prob in prob_list]

                    if len(self.fix_sentence) % 1000 == 0:
                        tqdm.write(f"Осталось предложений: {sum(prob_list)}")

                lang = random.choices(list(dict_langs.keys()), weights=random_prob_list)[0]
                a = 1

                langs_len_prefix = dict_langs_len_prefix[lang]

                step_view += 1
                sentence = []
                sentence_len_source = 0
                sentence_len_target = 0
                while True:
                    current_part = 0
                    is_empty = True
                    list_keys = list(dict_langs[lang].keys())
                    for part in list_keys:
                        if len(dict_langs[lang][part]) == 0:
                            dict_langs[lang].pop(part)
                        else:
                            current_part = part
                            is_empty = False

                    if is_empty:
                        break
                    
                    choise_idx = random.randint(0,len(dict_langs[lang][current_part])-1)
                    choise_splitter = random.choices(list_splitters_keys)[0]
                    splitter_len = len(list_splitters[choise_splitter])
                    idx = dict_langs[lang][current_part][choise_idx]
                    src_len, tgt_len = self._get_source_target_len(idx)
                    target = False #Ключ отвечает что текст target будет и в source

                    if random.randint(0,1000) == 555: #Низкая вероятность 1/1000
                        target = True

                    iter_langs_len_prefix = 0 #В начале source есть task: translate to:
                    if sentence_len_source != 0:
                        sentence_len_source -= langs_len_prefix
                        iter_langs_len_prefix = langs_len_prefix

                    a = 1
                    if target == False and sentence_len_source + splitter_len + src_len < max_len \
                        and sentence_len_target + splitter_len + tgt_len < max_len:
                        sentence.append((idx,iter_langs_len_prefix,target,choise_splitter))
                        sentence_len_source += splitter_len + src_len - 1
                        sentence_len_target += splitter_len + tgt_len - 1
                        del dict_langs[lang][current_part][choise_idx]
                    # elif target == False and sentence_len_source == 0 \
                    #     and sentence_len_target == 0 and \
                    #     (splitter_len + src_len >= max_len or splitter_len + tgt_len >= max_len):
                    #     #Частный случай когда строка длиннее
                    #     sentence.append((idx,iter_langs_len_prefix,target,choise_splitter))
                    #     sentence_len_source += src_len - 1
                    #     sentence_len_target += tgt_len - 1
                    #     del dict_langs[lang][current_part][choise_idx]
                    elif target == True and sentence_len_source + splitter_len + tgt_len < max_len \
                        and sentence_len_target + splitter_len + tgt_len < max_len:
                        sentence.append((idx,iter_langs_len_prefix,target,choise_splitter))
                        sentence_len_source += splitter_len + tgt_len - 1
                        sentence_len_target += splitter_len + tgt_len - 1
                        del dict_langs[lang][current_part][choise_idx]
                    else:
                        break

                if len(sentence) > 0:
                    self.fix_sentence.append(sentence)     
                    #self.fix_sentence_len.append((sentence_len_source,sentence_len_target))
            
            pickle.dump(self.fix_sentence, open(self.filename_len_index, 'wb'))

            a = 1
        

    def __getitem__(self,idx):
        dict_result = 0
        sentences = self.fix_sentence[idx]
        for sentence in sentences:
            (p_idx,iter_langs_len_prefix,target,choise_splitter) = sentence
            p_result = super().__getitem__(p_idx)
            if dict_result == 0:
                dict_result = p_result
            else:
                dict_result['input_ids'] = dict_result['input_ids'][:-1]
                dict_result['labels'] = dict_result['labels'][:-1]
                # print('======')
                # print(dict_result['input_ids'].dtype)
                # print(type(self.list_splitters))
                # print(choise_splitter)
                # print(self.list_splitters[choise_splitter])

                # self.list_splitters[choise_splitter]['input_ids'] = np.array(self.list_splitters[choise_splitter]['input_ids'])

                add_split = np.array(self.list_splitters[choise_splitter]['input_ids'],dtype=dict_result['input_ids'].dtype)

                dict_result['input_ids'] = np.concatenate((dict_result['input_ids'],add_split))
                dict_result['labels'] = np.concatenate((dict_result['labels'],add_split))

                if target:
                    dict_result['input_ids'] = np.concatenate((dict_result['input_ids'],p_result['labels']))
                    dict_result['labels'] = np.concatenate((dict_result['labels'],p_result['labels']))
                else:
                    dict_result['input_ids'] = np.concatenate((dict_result['input_ids'],p_result['input_ids'][iter_langs_len_prefix:]))
                    dict_result['labels'] = np.concatenate((dict_result['labels'],p_result['labels']))

        dict_result['len_input_ids'] = dict_result['input_ids'].shape[0]
        dict_result['len_labels'] = dict_result['labels'].shape[0]
        
        return dict_result

    def __len__(self):
        return len(self.fix_sentence)


# from transformers import T5Tokenizer
# max_input_length = 1024
# tokenizer = T5Tokenizer.from_pretrained('/data/tokenizers/tokenizer_t5_en_ru_zh_65000', max_len=max_input_length)
# # data_dir = '/share/datasets_bin/en_ru_zh_translate_corpus_freq_2'
# # rd_val = ReadDS(data_dir,'val',max_input_length)
# # print(len(rd_val))

# data_dir = '/share/datasets_bin/en_ru_zh_translate_corpus_freq_1'
# rd_val = FixLenFastReadDS(data_dir,'val',max_input_length,tokenizer)
# print(len(rd_val))
# print(rd_val[2])
