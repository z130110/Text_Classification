import torch 
from torch import nn
import numpy as np


class model_net(nn.Module):
	def __init__(self, arg_parser):
		super(model_net, self).__init__()
		np_pretrain_embeddings = np.load(arg_parser.data_dir + arg_parser.pretrain_subset).astype('float32')
		pretrain_embeddings = torch.tensor(np_pretrain_embeddings).float()
		self.pretrain_embeddings = nn.Embedding.from_pretrained(pretrain_embeddings, freeze = arg_parser.freeze_embed)
		self.embedding_dim = np_pretrain_embeddings.shape[1]
		self.kernal_sizes = [2,3,4]
		self.num_filters = 100
		self.leaky_relu = nn.LeakyReLU(0.1)
		self.num_class = arg_parser.num_class
		self.seq_length = arg_parser.seq_length		# 20
		self.conv_module = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.num_filters, k_size) for k_size in self.kernal_sizes])
		# for k_size in self.conv1d_kernal_sizes:
		# 	self.conv_module.append(nn.Conv1d(self.embedding_dim, self.num_filters, k_size))	
		self.feat_dim = self.num_filters * len(self.kernal_sizes)
		self.pool_module = nn.ModuleList([nn.MaxPool1d(self.seq_length - k_size + 1) for k_size in self.kernal_sizes]) 
		self.dropout = nn.Dropout(arg_parser.dropout)
		self.fully_connect = nn.Linear(self.feat_dim, self.num_class)
		

	def conv_pool(self, x, conv, pool):
		x = conv(x)			# [batch_size, num_filters, embed_dim - k_size + 1]
		x = self.leaky_relu(x)	# same shape as above
		x = pool(x)	# [batch_size, num_filters, 1]
		x = x.squeeze(2)		# [batch_size, num_filters]
		return x

	def forward(self, x): 	# input dim: [batch_size, padded_length]
		x = self.pretrain_embeddings(x)		# embedding layer, [batch_size, padded_length, embed_dim]
		x = torch.transpose(x, 2, 1) 	# torch 的 1dconv 输入纬度为 (batch_size, emb_dim, padded_length)
		kernal_feat = []
		for conv, pool in zip(self.conv_module, self.pool_module):
			conv_x = self.conv_pool(x, conv, pool)	# [batch_size, num_filters]
			kernal_feat.append(conv_x)		
		x = torch.cat(kernal_feat, 1) # 最后的句子特征向量 [batch_size, num_filters * num_kernal_size]
		x = self.dropout(x)
		#feat_dim = self.num_filters * len(self.conv1d_kernal_sizes)
		x = self.fully_connect(x)	   #  [batch_size, num_class]
		return x

