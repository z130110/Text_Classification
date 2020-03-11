import torch 
from torch import nn
import numpy as np
from pytorch_pretrained.modeling import BertModel

class model_net(nn.Module):
	def __init__(self, arg_parser):
		super(model_net, self).__init__()
		self.num_class = arg_parser.num_class
		self.bert_pretrain_path = arg_parser.bert_pretrain_path
		self.bert_model = BertModel.from_pretrained(self.bert_pretrain_path)
		self.fully_connect = nn.Linear(768, self.num_class)
		for layer in self.bert_model.parameters():
			layer.requires_grad = arg_parser.bert_gradient

	def forward(self, x): 
		input_x, masks = x[0], x[1]
		bert_out, pooled = self.bert_model(input_x, attention_mask=masks, output_all_encoded_layers=False)
		# bert_out dim: [batch size, hidden size]
		out = self.fully_connect(pooled)
		return out

