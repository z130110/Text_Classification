import torch 
from torch import nn
import numpy as np
from pytorch_pretrained.modeling import BertModel

class model_net(nn.Module):
	def __init__(self, arg_parser):
		super(model_net, self).__init__()
		self.num_class = arg_parser.num_class
		self.seq_len = arg_parser.seq_length
		self.bert_pretrain_path = arg_parser.bert_pretrain_path
		self.bert_model = BertModel.from_pretrained(self.bert_pretrain_path)
		for layer in self.bert_model.parameters():
			layer.requires_grad = arg_parser.bert_gradient
		self.hidden_size_bert = 768	
		self.hidden_size_lstm = 64
		self.BiLSTM = nn.LSTM(self.hidden_size_bert, self.hidden_size_lstm, num_layers = 2, batch_first = True, \
					dropout = arg_parser.dropout, bidirectional = True)
		self.fully_connect = nn.Linear(self.hidden_size_lstm * 2, self.num_class)
 
	def forward(self, x): 
		input_x, masks = x[0], x[1]
		last_encoding, pooled = self.bert_model(input_x, attention_mask=masks, output_all_encoded_layers=False)
		# last encoding dim: [batch size, seq len, hidden size 768]			
		out, h_c = self.BiLSTM(last_encoding) # out: [batch_size, seq_length, hidden_size * 2], h_c: (hidden_, cell_)
		last_hidden = out[:, -1, :]
		out = self.fully_connect(last_hidden)
		return out

