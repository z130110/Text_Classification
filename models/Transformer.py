import torch 
from torch import nn
import numpy as np
import torch.nn.functional as F
import copy

class model_net(nn.Module):
	def __init__(self, arg_parser):
		super(model_net, self).__init__()
		np_pretrain_embeddings = np.load(arg_parser.pretrain_subset).astype('float32')
		pretrain_embeddings = torch.tensor(np_pretrain_embeddings).float()
		self.pretrain_embeddings = nn.Embedding.from_pretrained(pretrain_embeddings, freeze = arg_parser.freeze_embed)
		self.vocab_size = np_pretrain_embeddings.shape[0]
		self.embedding_dim = np_pretrain_embeddings.shape[1]
		#self.pretrain_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.leaky_relu = nn.LeakyReLU(0.1)
		self.num_class = arg_parser.num_class
		self.seq_length = arg_parser.seq_length
		self.batch_size = arg_parser.batch_size
		self.num_encoder = 1
		self.dim_model = self.embedding_dim
		self.positional_encoding = positional_encoding(self.seq_length, self.dim_model, arg_parser.dropout)
		self.encoder = Encoder(dim_model = self.dim_model)
		self.encoder_list = nn.ModuleList(copy.deepcopy(self.encoder) for i in range(self.num_encoder))
		self.fc_last = nn.Linear(self.seq_length * self.embedding_dim, self.num_class)

	def forward(self, x):

		x = self.pretrain_embeddings(x)
		x = self.positional_encoding(x)
		for encoder in self.encoder_list:
			x = encoder(x)
		x = x.view(self.batch_size, self.seq_length * self.embedding_dim) 
		x = self.fc_last(x)
		return x 

class positional_encoding(nn.Module):
	def __init__(self, seq_len, dim_model, dropout = 0.5):
		super(positional_encoding, self).__init__()
		self.pe = np.array([[pos/(10000 ** (2*i/dim_model)) for i in range(1, dim_model+1)] for pos in range(1, seq_len + 1)])
		self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
		self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
		self.pe = torch.from_numpy(self.pe).float()
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = x + nn.Parameter(self.pe)
		out = self.dropout(out)
		return out

class Encoder(nn.Module):
	def __init__(self, dim_model):
		super(Encoder, self).__init__()
		self.attention = multi_head_attention(dim_model = dim_model)
		self.feed_forward = Position_wise_feed_forward(dim_model = dim_model)
	
	def forward(self, x):
		out = self.attention(x)
		out = self.feed_forward(out)
		return out


class multi_head_attention(nn.Module):
	def __init__(self, dim_model, num_head = 5):
		super(multi_head_attention, self).__init__()
		self.num_head = num_head	# 5
		self.dim_model = dim_model	# 300
		self.dim_head = self.dim_model // self.num_head
		self.fc_V = nn.Linear(self.dim_model, self.num_head * self.dim_head)
		self.fc_K = nn.Linear(self.dim_model, self.num_head * self.dim_head)
		self.fc_Q = nn.Linear(self.dim_model, self.num_head * self.dim_head)
		self.attention = scaled_dot_product_attention()
		self.fc_head = nn.Linear(num_head * self.dim_head, dim_model)
		self.dropout_head = nn.Dropout(0.5)
		self.layernorm_attention = nn.LayerNorm(dim_model)

	def forward(self, x):
		batch_size = x.shape[0]
		seq_len = x.shape[1]
		Q = self.fc_Q(x)		# [batch_size, seq_len, dim_model]
		K = self.fc_K(x)		# [batch_size, seq_len, dim_model]
		V = self.fc_V(x)		# [batch_size, seq_len, dim_model]
		Q = Q.view(batch_size, -1, self.num_head, self.dim_head)	# [batch_size, seq_len, num_head, dim_head]
		K = K.view(batch_size, -1, self.num_head, self.dim_head)	# [batch_size, seq_len, num_head, dim_head]
		V = V.view(batch_size, -1, self.num_head, self.dim_head)	# [batch_size, seq_len, num_head, dim_head]
		Q = Q.transpose(1,2)	# [batch_size, num_head, seq_len, dim_head]
		K = K.transpose(1,2)	# [batch_size, num_head, seq_len, dim_head]
		V = V.transpose(1,2)	# [batch_size, num_head, seq_len, dim_head]
		attention_score = self.attention(Q, K, V)
		attention_score = attention_score.transpose(1,2).reshape(batch_size, seq_len, self.num_head * self.dim_head)
		out = self.fc_head(attention_score)
		out = self.dropout_head(out)
		out = x + out
		out = self.layernorm_attention(out)
		return out

class scaled_dot_product_attention(nn.Module):
	def __init__(self, scale = False, dropout = 0.5):
		super(scaled_dot_product_attention, self).__init__()
		self.dropout_scale = nn.Dropout(dropout)
		self.scale = scale

	def forward(self, Q, K, V):
		attention = torch.matmul(Q, K.transpose(3,2))
		if self.scale == True:
			attention = attention * scale
		attention = self.dropout_scale(attention)
		attention = F.softmax(attention, dim = -1)
		attention_score = torch.matmul(attention, V)
		return attention_score

class Position_wise_feed_forward(nn.Module):
	def __init__(self, dim_model, dim_hidden = 512, dropout = 0.5):
		super(Position_wise_feed_forward, self).__init__()
		self.fc_1 = nn.Linear(dim_model, dim_hidden)
		self.fc_2 = nn.Linear(dim_hidden, dim_model)
		self.dropout = nn.Dropout(0.5)
		self.layernorm_feed_forward = nn.LayerNorm(dim_model)

	def forward(self, x):
		out = F.relu(self.fc_1(x))
		#out = F.leaky_relu(self.fc_1(x), negative_slope = 0.1)
		out = self.fc_2(out)
		out = self.dropout(out)
		out = x + out
		out = self.layernorm_feed_forward(out)
		return out

