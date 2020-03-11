import torch
import numpy as np
import jieba
import pandas as pd
from tqdm import tqdm
import itertools
import os 
import json
import pickle
from pytorch_pretrained.tokenization import BertTokenizer
from pytorch_pretrained.modeling import BertModel


class processor(object):
	def __init__(self, arg_parser):
		#self.data_dir = data_dir
		self.arg_parser = arg_parser
		self.train_text = arg_parser.data_dir + arg_parser.train_text
		self.dev_text = arg_parser.data_dir + arg_parser.dev_text
		self.test_text = arg_parser.data_dir + arg_parser.test_text
		self.pretrain_subset = arg_parser.data_dir + arg_parser.pretrain_subset
		self.index_2vocab_file = arg_parser.data_dir + "index_2vocab_file.json"
		self.vocab_2index_file = arg_parser.data_dir + "vocab_2index_file.json"
		self.bijection_json = arg_parser.data_dir + "ori_new_bijection.json"
		self.UNK, self.PAD = "+", "~" # 未知字和padding符号 '<UNK>' '<PAD>',选取预训练中的2个冷门符号词向量代替"+" "&"
		self.pad_size = arg_parser.seq_length 	# 小于20的句子padding, 多余20的cut处理
		self.ascii_remove = list(range(46, 58)) + list(range(64, 126))
		self.train_reindex = arg_parser.data_dir + arg_parser.train_reindex
		self.dev_reindex = arg_parser.data_dir + arg_parser.dev_reindex
		self.test_reindex = arg_parser.data_dir + arg_parser.test_reindex
		# self.train_reindex = arg_parser.data_dir + "train_index.npy"
		# self.dev_reindex = arg_parser.data_dir + "dev_index.npy"
		# self.test_reindex = arg_parser.data_dir + "test_index.npy"

	def get_data(self):
		train_bool = os.path.exists(self.train_reindex)
		dev_bool = os.path.exists(self.dev_reindex)
		test_bool = os.path.exists(self.test_reindex)
		pretrained_bool = os.path.exists(self.pretrain_subset)
		if train_bool and dev_bool and dev_bool and pretrained_bool:
			np_train_data = np.load(self.train_reindex, allow_pickle = True)
			np_dev_data = np.load(self.dev_reindex, allow_pickle = True)
			np_test_data = np.load(self.test_reindex, allow_pickle = True)
			np_pretrained_subset = np.load(self.pretrain_subset).astype('float32')
			return np_train_data, np_dev_data, np_test_data, np_pretrained_subset
		else:
			np_train_contents, np_dev_contents, np_test_contents, np_embeddings_subset = self.index_vocabulary()
			return np_train_contents, np_dev_contents, np_test_contents, np_embeddings_subset

	def index_vocabulary(self):
		index_to_vocab, vocab_to_index, vocab_embed_dico = self.build_vocab_embed()
		bijection, sorted_unique_ori = self.build_cardinarity_map()
		embeddings_subset = []
		for index in sorted_unique_ori:
			vocab = index_to_vocab.get(index)
			embedding = vocab_embed_dico[vocab]
			embeddings_subset.append(embedding)
		np_embeddings_subset = np.array(embeddings_subset)
		np.save(self.pretrain_subset, np_embeddings_subset)
		def re_index(contents, bijection = bijection):
			for i in tqdm(range(len(contents))):					
				sentence = contents[i][0]
				new_sentence = [bijection[word] for word in sentence]
				contents[i][0] = new_sentence
			return contents
		new_train, new_dev, new_test = re_index(self.train_contents), re_index(self.dev_contents), re_index(self.test_contents)
		np_train_contents = np.array(new_train)
		np_dev_contents = np.array(new_dev)
		np_test_contents = np.array(new_test)
		np.save(self.train_reindex, np_train_contents)
		np.save(self.dev_reindex, np_dev_contents)
		np.save(self.test_reindex, np_test_contents)
		with open(self.index_2vocab_file, "w") as index_2vocab_obj:
			json.dump(index_to_vocab, index_2vocab_obj) 
		with open(self.vocab_2index_file, "w") as vocab_2index_obj:
			json.dump(vocab_to_index, vocab_2index_obj) 
		return np_train_contents, np_dev_contents, np_test_contents, np_embeddings_subset

	def build_vocab_embed(self):
		# 建立预训练词向量中词和index的整数对应, 从0开始索引
		with open(self.arg_parser.public_pretrain, "r") as embed_obj:
			index_to_vocab, vocab_to_index, vocab_embed_dico = {}, {}, {}
			for i, line in enumerate(tqdm(embed_obj)):
				line = line.split()
				vocab = line.pop(0)
				embed = [float(e) for e in line]
				vocab_embed_dico[vocab] = embed
				index_to_vocab[i] = vocab
				vocab_to_index[vocab] = i
		self.index_to_vocab, self.vocab_to_index, self.vocab_embed_dico = index_to_vocab, vocab_to_index, vocab_embed_dico
		return index_to_vocab, vocab_to_index, vocab_embed_dico
	
	def build_cardinarity_map(self):
		# [([句子2list], label,  序列长度), ([句子2list], label,  序列长度), ...]
		self.train_contents = self.token_words(self.train_text)
		self.dev_contents = self.token_words(self.dev_text)
		self.test_contents = self.token_words(self.test_text)
		def word_unique(contents):	# 获得word cardinarity
			collect_card = []
			for sentence_tuple in tqdm(contents):
				sentence = sentence_tuple[0]
				for word in sentence:
					if word not in collect_card:
						collect_card.append(word)
			return collect_card
		train_card, dev_card, test_card = word_unique(self.train_contents), word_unique(self.dev_contents), word_unique(self.test_contents)
		sort_all_unique = sorted(set(train_card + dev_card + test_card))
		ordered_unique = list(range(len(sort_all_unique)))
		bijection = dict(zip(sort_all_unique, ordered_unique))
		with open(self.bijection_json, "w") as bijection_obj:
			json.dump(bijection, bijection_obj)
		return bijection, sort_all_unique  # {original index: new index}

	def token_words(self, text_file):
		# 文件格式每行为: '体验2D巅峰 倚天屠龙记十大创新概览\t8\n', 
		# 				'60年铁树开花形状似玉米芯(组图)\t5\n'
		# 其中最后的数字为标签，去除 \t \n, 取出标签
		# 返回格式: [([句子2list], label,  序列长度), ([句子2list], label,  序列长度), ...]
		#index_to_vocab, vocab_to_index = self.build_vocab_map()		
		with open(text_file, "r") as text_obj:
			contents = []
			for line in tqdm(list(text_obj)):  
				line = line.replace("\t", "").replace("\n", "")  #去除 "\t" "\n"
				line = line.split()
				label = int(line.pop())		# 去除标签
				line = "".join(line)
				seg_list = list(jieba.cut(line))  # 结巴分词
				seg_list = [e for e in seg_list if e != " "]  # 移除空格字符 " "
				inexist = []    # pre_trained 中没有的词，拆成独立的字符
				sentence = []
				for e in seg_list:
					retrive_index = self.vocab_to_index.get(e)
					if retrive_index == None:  # 如果不在预训练中，长度小于1为unknow, 大于等于一过滤掉数字和字母的组合
						split_e = list(e)
						if len(split_e) == 1:
							sentence.append(self.vocab_to_index.get(self.UNK))
						else:
							chars = [i for i in split_e if ord(i) not in self.ascii_remove]
							char_indices = [self.vocab_to_index[char] for char in chars if self.vocab_to_index.get(char) != None]
							sentence = list(itertools.chain(sentence, char_indices))
					else:
						sentence.append(retrive_index)				
				if len(sentence) >= self.pad_size:
					sentence = sentence[:self.pad_size]
					seq_len = self.pad_size
				else:
					seq_len = len(sentence)
					num_pad = self.pad_size - seq_len
					sentence = sentence + [self.vocab_to_index.get(self.PAD)] * num_pad 
				contents.append([sentence,label,seq_len])
		return contents

class processor_bert(object):
	def __init__(self, arg_parser):
		#self.data_dir = data_dir
		self.arg_parser = arg_parser
		self.train_text = arg_parser.data_dir + arg_parser.train_text
		self.dev_text = arg_parser.data_dir + arg_parser.dev_text
		self.test_text = arg_parser.data_dir + arg_parser.test_text
		self.bert_path = arg_parser.bert_pretrain_path
		self.CLS, self.SEP, self.PAD = ["[CLS]"], ["[SEP]"], ["[PAD]"] 
		self.pad_size = arg_parser.seq_length 		# 小于seq len的句子padding, 多余的cut处理
		#self.ascii_remove = list(range(46, 58)) + list(range(64, 126))
		self.model_type = arg_parser.bert_pretrain_path.split("_")[0]
		self.train_bert = arg_parser.data_dir + "train_" + self.model_type + "_" + str(self.pad_size) + ".npy"
		self.dev_bert = arg_parser.data_dir + "dev_"  + self.model_type + "_" + str(self.pad_size) + ".npy"
		self.test_bert = arg_parser.data_dir + "test_" + self.model_type + "_" + ".npy"
		self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

	def get_data(self):
		train_bool = os.path.exists(self.train_bert)
		dev_bool = os.path.exists(self.dev_bert)
		test_bool = os.path.exists(self.test_bert)
		if train_bool and dev_bool and dev_bool:
			np_train_data = np.load(self.train_bert, allow_pickle = True)
			np_dev_data = np.load(self.dev_bert, allow_pickle = True)
			np_test_data = np.load(self.test_bert, allow_pickle = True)
			return np_train_data, np_dev_data, np_test_data
		else:
			np_train_data, np_dev_data, np_test_data = self.save_train_data()
			return np_train_data, np_dev_data, np_test_data

	def save_train_data(self):
		train_bert_np = np.array(self.token_word_bert(self.train_text))
		dev_bert_np = np.array(self.token_word_bert(self.dev_text))
		test_bert_np = np.array(self.token_word_bert(self.test_text))
		np.save(self.train_bert, train_bert_np)
		np.save(self.dev_bert, dev_bert_np)
		np.save(self.test_bert, test_bert_np)
		return train_bert_np, dev_bert_np, test_bert_np

	def token_word_bert(self, text_file):
		# 文件格式每行为: '体验2D巅峰 倚天屠龙记十大创新概览\t8\n', 
		# 				'60年铁树开花形状似玉米芯(组图)\t5\n'
		# 其中最后的数字为标签，去除 \t \n, 取出标签
		# return: [token ids, label, seq_len, mask]
		#text_file = self.train_text
		with open(text_file, "r") as text_obj:
			contents = []
			counter = 0
			for line in tqdm(list(text_obj)):
				line = line.replace("\t", "").replace("\n", "")  #去除 "\t" "\n"
				label = int(line[-1])
				line = line[:-1]  # 去除标签
				token = self.tokenizer.tokenize(line)
				if len(token) >= self.pad_size - 2:
					token = self.CLS + self.tokenizer.tokenize(line)[:self.pad_size - 2] + self.SEP
					seq_len = self.pad_size - 2
					mask = [1] * self.pad_size
				else:
					seq_len = len(token)
					token = self.CLS + self.tokenizer.tokenize(line) + self.SEP
					mask = len(token) * [1] + (self.pad_size - len(token)) * [0]
					token = token + (self.pad_size - len(token)) * self.PAD					
				token_ids = self.tokenizer.convert_tokens_to_ids(token)
				contents.append([token_ids, label, seq_len, mask])
		return contents


class build_batch_iter(object):
	def __init__(self, arg_parser, contents, device):
		# 数据格式为普通list [[sentence1], label1, seq_len1], [sentence2], label2, seq_len2], ....]
		self.arg_parser = arg_parser
		self.contents = contents
		self.batch_size = self.arg_parser.batch_size	# default 128
		self.num_samples = len(contents)
		self.num_minibatch = self.num_samples // self.batch_size
		self.residue = self.num_samples % self.num_minibatch
		self.batch_index = 0
		self.device = device

	def to_tensor(self, contents):
		sentences = torch.LongTensor([instance[0] for instance in contents]).to(self.device)
		labels = torch.LongTensor([instance[1] for instance in contents]).to(self.device)
		seq_lens = torch.LongTensor([instance[2] for instance in contents]).to(self.device)	
		if self.arg_parser.bert:
			masks = torch.LongTensor([instance[3] for instance in contents]).to(self.device)
			return (sentences, labels, masks)
		else: 
			return (sentences, labels)

	def __next__(self):
		if self.batch_index < self.num_minibatch:
			batch_data = self.contents[self.batch_size * self.batch_index : self.batch_size * (self.batch_index + 1)]
			self.batch_index += 1
			batch_data = self.to_tensor(batch_data)
			return batch_data
		elif self.residue != 0 and self.batch_index == self.num_minibatch:
			batch_data = self.contents[self.batch_size * self.batch_index:]
			self.batch_index += 1
			batch_data = self.to_tensor(batch_data)
			return batch_data			
		else:
			self.batch_index = 0
			raise StopIteration

	def __iter__(self):	
		return self

	def __len__(self):
		return self.num_minibatch if self.residue == 0 else self.num_minibatch + 1

	def __getitem__(self, position):
		if self.residue == 0 and position < self.num_minibatch:
			batch_data = self.contents[self.batch_size * position : self.batch_size * (position + 1)]
			batch_data = self.to_tensor(batch_data)
			return batch_data
		elif self.residue != 0 and position < self.num_minibatch:
			batch_data = self.contents[self.batch_size * position : self.batch_size * (position + 1)]
			batch_data = self.to_tensor(batch_data)
			return batch_data
		elif self.residue != 0 and position == self.num_minibatch:
			return self.to_tensor(self.contents[self.batch_size * position : ])
		else:
			raise IndexError(": Index out of range")	

