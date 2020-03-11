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
		self.kernal_sizes = [2,3,4]
		self.num_filters = 100
		self.leaky_relu = nn.LeakyReLU(0.1)
		self.conv_module = nn.ModuleList([nn.Conv1d(self.hidden_size_bert, self.num_filters, ks) for ks in self.kernal_sizes]) 
		self.pool_module = nn.ModuleList([nn.MaxPool1d(self.seq_len - ks + 1) for ks in self.kernal_sizes])
		self.dropout = nn.Dropout(arg_parser.dropout)
		self.feat_dim = len(self.kernal_sizes) * self.num_filters
		self.fully_connect = nn.Linear(self.feat_dim, self.num_class)

	def forward(self, x): 
		input_x, masks = x[0], x[1]
		last_encoding, pooled = self.bert_model(input_x, attention_mask=masks, output_all_encoded_layers=False)
		# last encoding dim: [batch size, seq len, hidden size 768]			
		x = torch.transpose(last_encoding, 2, 1)	# torch 的 1dconv 输入纬度为 (batch_size, emb_dim, padded_length)
		kernal_feat = []
		for conv, pool in zip(self.conv_module, self.pool_module):
			conv_out = conv(x)
			relu_out = self.leaky_relu(conv_out)
			pool_out = pool(relu_out).squeeze(2)
			kernal_feat.append(pool_out)
		concat_feat = torch.cat(kernal_feat,1)
		out = self.dropout(concat_feat)
		out = self.fully_connect(out)
		return out

