import torch 
from torch import nn
import numpy as np

'''Recurrent Convolutional Neural Networks for Text Classification'''
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
		self.hidden_size = arg_parser.hidden_size
		self.BiLSTM = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers = 2, batch_first = True, \
							dropout = arg_parser.dropout, bidirectional = True)
		self.conv_module = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.num_filters, k_size) for k_size in self.kernal_sizes])
		self.tanh = nn.Tanh()
		self.maxpool = nn.MaxPool1d(self.seq_length)
		self.fully_connect = nn.Linear(self.embedding_dim + self.hidden_size * 2, self.num_class)
		
	def forward(self, x): 	# input dim: [batch_size, padded_length]
		embed = self.pretrain_embeddings(x)		# embedding layer, [batch_size, padded_length, embed_dim]
		last_hidden, h_c = self.BiLSTM(embed)		# x: [batch_size, seq_length, hidden_size * 2], h_: [1, batch_size, hidden_size * 2]
		out = torch.cat([embed,last_hidden], 2)		# [batch size, seq len, embed dim + hidden size * 2]
		out = self.tanh(out)
		out = out.permute(0,2,1)				
		out = self.maxpool(out).squeeze(2)		# pooling along with sequence length's dimension.
		out = self.fully_connect(out)
		return out


