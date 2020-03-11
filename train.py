import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import os
from time import time
import datetime
from logging import getLogger
from pytorch_pretrained.optimization import BertAdam

logger = getLogger("my logger")
#torch.set_printoptions(precision = 3, sci_mode = False)

class trainer(object):
	def __init__(self, model, arg_parser, train_batch, dev_batch, test_batch, model_save_path, torch_device):
		self.model = model
		self.lr = arg_parser.lr
		self.lr_decay = arg_parser.lr_decay
		self.arg_parser = arg_parser
		self.lr_patience = arg_parser.lr_patience
		self.init_method = arg_parser.init_method
		self.torch_device = torch_device
		self.train_batch = train_batch
		self.test_batch = test_batch
		self.dev_batch = dev_batch
		self.model_save_path = model_save_path
		self.iter_patience = 0
		self.epoch_patience = 0
		self.EarlyStopping_triggered = False
		self.unbalanced = arg_parser.unbalanced_distribution
		self.class_names = [x.strip() for x in open(arg_parser.data_dir + arg_parser.class_name, encoding='utf-8').readlines()]
		#self.class_num = 22
		if not self.arg_parser.bert:
			self.init_weights_()

	def init_weights_(self, layer_name_1 = "embedding", layer_name_2 = "norm"):
		init_list = ["uniform_", "normal_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", \
					"kaiming_normal_", "orthogonal_"]
		for name, weight in self.model.named_parameters():
			if layer_name_1 not in name and layer_name_2 not in name:
				if "weight" in name:
					getattr(nn.init,self.init_method)(weight)
				elif "bias" in name:
					getattr(nn.init,"constant_")(weight, 0.1)
				else:
					pass

	def train_(self):
		batch_size = self.arg_parser.batch_size
		epochs = self.arg_parser.num_epochs
		self.model.train()
		iter_counter = 0
		# Adadelta, Adagrad, Adam, AdamW, SGD, SparseAdam, Adamax, ASGD, RMSprop, LBFGS, Rprop
		best_dev_loss = float("inf")
		best_dev_accuracy = 0 
		train_loss_info = {} 		# collection loss data to draw the loss curve
		train_loss_info["num_epochs"] = epochs
		train_loss_info["batch_size"] = batch_size
		if self.arg_parser.bert:
			param_optimizer = list(self.model.named_parameters())
			no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
				{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
			optimizer = BertAdam(optimizer_grouped_parameters, lr = self.lr, warmup=0.05, t_total= epochs * len(self.train_batch))
		else:
			optimizer = getattr(torch.optim, self.arg_parser.optimizer)(self.model.parameters(), lr = self.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = self.lr_decay, patience = self.lr_patience, verbose = True)
		for epoch in range(epochs):
			logger.info(f"-" * 35 + ">" + f"Training {epoch}th epoch" + "<" + "-"*35)
			#optimizer = getattr(torch.optim, self.arg_parser.optimizer)(self.model.parameters(), lr = self.lr * (self.lr_decay ** epoch))
			last_epoch_dev_loss = float("inf") if epoch == 0 else epoch_dev_loss
			last_epoch_dev_accuracy = 0 if epoch == 0 else epoch_dev_accuracy
			epoch_train_loss = 0; epoch_train_accuracy = 0	
			epoch_dev_loss = 0; epoch_dev_accuracy = 0	
			print_counter = 0
			for minibatch in self.train_batch:
				if self.arg_parser.bert:
					input_x = minibatch[0]; masks = minibatch[2]
					input_ = (input_x, masks)
				else:
					input_ = minibatch[0]
				label_ = minibatch[1]
				output_ = self.model(input_)
				self.model.zero_grad()
				loss = F.cross_entropy(output_, label_)
				loss.backward()
				optimizer.step()
				iter_counter += 1
				if iter_counter % 100 == 0:
					predict = output_.max(1)[1].cpu()
					label_cpu = label_.cpu()
					train_loss = loss.cpu().item()
					train_loss = round(train_loss, 5)
					#train_accuracy = round(accuracy_score(label_cpu, predict), 4)
					train_accuracy = accuracy_score(label_cpu, predict)
					dev_loss, dev_accuracy, dev_f1_macro, dev_f1_micro, dev_weighted = self.evaluation(self.model, self.dev_batch)
					epoch_train_loss += train_loss; epoch_train_accuracy += train_accuracy; 
					epoch_dev_loss += dev_loss; epoch_dev_accuracy += dev_accuracy
					logger.info(f"Iter: {iter_counter}, train loss: {train_loss}, train accuracy: {train_accuracy}, val loss: {dev_loss}, val accuracy: {dev_accuracy}")	
					if self.unbalanced == True:
						logger.info(f"val F1 macro: {dev_f1_macro}, val F1 micro: {dev_f1_micro}, val F1 weighted: {dev_weighted}")
					self.model.train()
					if dev_loss < best_dev_loss and dev_accuracy > best_dev_accuracy:
						best_dev_loss = dev_loss
						best_dev_accuracy = dev_accuracy
						#logger.info(f"Best validation loss updated: {best_dev_loss}")
						torch.save(self.model, self.model_save_path)
					else:
						self.iter_patience += 1	
					print_counter += 1
			epoch_train_loss = round(epoch_train_loss/print_counter, 5)
			epoch_train_accuracy = round(epoch_train_accuracy/print_counter, 5)
			epoch_dev_loss = round(epoch_dev_loss/print_counter, 5)
			epoch_dev_accuracy = round(epoch_dev_accuracy/print_counter, 5)
			scheduler.step(epoch_dev_loss)
			logger.info(f"{epoch}th epoch finished, val epoch loss:{epoch_dev_loss}, val epoch accuracy:{epoch_dev_accuracy}")
			self.EarlyStopping(epoch_dev_loss, epoch_dev_accuracy, last_epoch_dev_loss, last_epoch_dev_accuracy)
			if self.EarlyStopping_triggered == True:
				logger.info("=" * 70)	
				logger.info(f"Early Stopping triggered after {epoch + 1} epoches, calculating test accuracy...")	
				break
		if self.EarlyStopping_triggered == False:	
			logger.info("Training fnished, full epoch, calculating test accuracy...")
		self.evaluate_test()				

	def EarlyStopping(self, cur_dev_loss, cur_dev_accuracy, last_dev_loss, last_dev_accuracy, monitor='dev_loss'):
		if monitor == "dev_loss": 
			self.epoch_patience += 1 if cur_dev_loss >= last_dev_loss else 0
		if monitor == "dev_accuracy": 
			self.epoch_patience += 1 if cur_dev_accuracy <= last_dev_accuracy else 0
		if self.epoch_patience == self.arg_parser.num_patience:
			torch.save(self.model, self.model_save_path)
			self.EarlyStopping_triggered = True		

	def evaluation(self, model, batches, test = False):
		model.eval()
		loss = 0
		label_all = np.array([], dtype = int)
		predict_all = np.array([], dtype = int)
		with torch.no_grad():
			num_iter = 0
			for patch in batches:
				if self.arg_parser.bert:
					input_x = patch[0]; masks = patch[2]
					input_ = (input_x, masks)
				else:
					input_ = patch[0]				
				label_ = patch[1]
				output_ = model(input_)
				loss += F.cross_entropy(output_, label_).cpu()
				predict = output_.max(1)[1].cpu()
				label_all = np.append(label_all, label_.cpu().numpy())
				predict_all = np.append(predict_all, predict.numpy())
				num_iter += 1
		# loss = loss / num_iter
		loss = round(loss.item(), 5)				
		accuracy = accuracy_score(label_all, predict_all)
		accuracy = round(accuracy, 4)
		if test == True:
			self.test_labels = label_all
			self.test_predict = predict_all
		f1_macro = f1_score(label_all, predict_all, average = "macro")
		f1_micro = f1_score(label_all, predict_all, average = "micro")
		f1_weighted = f1_score(label_all, predict_all, average = "weighted")
		f1_macro, f1_micro, f1_weighted = round(f1_macro, 5), round(f1_micro, 5),  round(f1_weighted, 5)
		return loss, accuracy, f1_macro, f1_micro, f1_weighted

	def evaluate_test(self):
		loaded_model = torch.load(self.model_save_path, map_location = self.torch_device)
		test_loss, test_accuracy, test_f1_macro, test_f1_micro, test_f1_weighted = self.evaluation(loaded_model, self.test_batch, test = True)
		logger.info(f"test loss:{test_loss}, test accuracy: {test_accuracy}")
		logger.info("Precision, Recall and F1-Score...")
		eval_report = classification_report(self.test_labels, self.test_predict, target_names = self.class_names, digits=4)
		logger.info(eval_report)
		logger.info("Confusion Matrix...")
		confusion = confusion_matrix(self.test_labels, self.test_predict)
		logger.info(confusion)







