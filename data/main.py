from transformers import T5Tokenizer
from read_data_fast import FixLenFastReadDS

max_input_length = 200
tokenizer = T5Tokenizer.from_pretrained('/data/tokenizers/tokenizer_t5_en_ru_zh_65000', max_len=max_input_length)
# data_dir = '/share/datasets_bin/en_ru_zh_translate_corpus_freq_2'
# rd_val = ReadDS(data_dir,'val',max_input_length)
# print(len(rd_val))

data_dir = '/share/datasets_bin/en_ru_zh_translate_corpus_freq_2'
rd_val = FixLenFastReadDS(data_dir,'train',max_input_length,tokenizer)
print(len(rd_val))
print(rd_val[2])
