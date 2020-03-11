import torch 
from torch import nn
import numpy as np


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
		self.hidden_size = arg_parser.hidden_size
		self.BiLSTM = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers = 2, batch_first = True, \
							dropout = arg_parser.dropout, bidirectional = True)
		self.fully_connect = nn.Linear(self.hidden_size * 2, self.num_class)	

	def forward(self, x):
		x = self.pretrain_embeddings(x)		# embedding layer, [batch_size, seq_length, embed_dim]
		x, h_c = self.BiLSTM(x)		# x: [batch_size, seq_length, hidden_size * 2], h_: [1, batch_size, hidden_size * 2]
		x = x[:, -1, :]						# last hidden state in the sentence
		output = self.fully_connect(x)		#  [batch_size, num_class]	
		return output





