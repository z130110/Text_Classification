import torch
import argparse
from torch import nn
from importlib import import_module
from train import trainer
from logger import create_logger
import datetime
import os
import ast

parser = argparse.ArgumentParser(description = "Chinese Text Classification in Practice")
parser.add_argument("--model", type = str, default = "TextCNN", help = "select a model")
parser.add_argument("--data_dir", type = str, default = "data/THUCNews/", help = "directory which storing data")
#parser.add_argument("--data_dir", type = str, default = "data/liquid/", help = "directory which storing data")
parser.add_argument("--unbalanced_distribution", type = ast.literal_eval, default = False, help = "Whether the dataset's distribution unbalanced or not")
parser.add_argument("--train_text", type = str, default = "train.txt", help = "training text file")
parser.add_argument("--dev_text", type = str, default = "dev.txt", help = "validation text file")
parser.add_argument("--test_text", type = str, default = "test.txt", help = "test text file")
parser.add_argument("--class_name", type = str, default = "class.txt", help = "test text file")
parser.add_argument("--num_class", type = int, default = 22)
parser.add_argument("--public_pretrain", type = str, default = "../public_pretrained/sgns.sogou.char", help = "open source public pretrained embeddings")
parser.add_argument("--pretrain_subset", type = str, default = "pretrained_subset.npy", help = "subset embeddings")
parser.add_argument("--train_reindex",type = str,  default = "train_index.npy", help = "new index for train contents")
parser.add_argument("--dev_reindex", type = str, default = "dev_index.npy", help = "new index for dev contents")
parser.add_argument("--test_reindex", type = str, default = "test_index.npy", help = "new index for test contents")
parser.add_argument("--num_train", type = int, default = 180000, help = "Number of training samples")
parser.add_argument("--num_validation", type = int, default = 10000, help = "Number of validation samples")
parser.add_argument("--num_test", type = int, default = 10000, help = "Number of test samples")
parser.add_argument("--embedding_dim", type = int, default = 300)
parser.add_argument("--hidden_size", type = int, default = 64, help = "hidden dimension when using RNN cells")
parser.add_argument("--dropout", type = float, default = 0.5)
parser.add_argument("--seq_length", type = int, default = 32, help = "number of fixed word for each sentence")
parser.add_argument("--device", type = str, default = "cpu", help = "GPU or CPU")
parser.add_argument("--batch_size", type = int, default = 100, help = "batch size")
parser.add_argument("--num_epochs", type = int, default = 64, help = "batch size")
parser.add_argument("--num_patience", type = int, default = 15, help = "number of times to trick early stopping")
parser.add_argument("--lr", type = float, default = 0.0005, help = "learning rate")
parser.add_argument("--lr_decay", type = float, default = 0.6, help = "learning rate")
parser.add_argument("--lr_patience", type = float, default = 2, help = "shrinking the lr if no update")
parser.add_argument("--optimizer", type = str, default = "Adam", help = "Optimizer's name, must be method of torch.optim")
parser.add_argument("--init_method", type = str, default = "xavier_normal_", help = "Method which initilize the net'weights ")
parser.add_argument("--freeze_embed", type = ast.literal_eval, default = True, help = "freeze the embedding layer or not")
parser.add_argument("--torch_seed", type = int, default = 1, help = "torch's random seed")
parser.add_argument("--bert", type = ast.literal_eval, default = False, help = "Bert or not")
parser.add_argument("--bert_gradient", type = ast.literal_eval, default = False, help = "Back propagate Bert gradients or not")
parser.add_argument("--bert_pretrain_path", type = str, default = "ERNIE_pretrain", help = "Bert pretrain path")
args = parser.parse_args()

if __name__ == "__main__":
	date_dir = datetime.datetime.now().strftime("%Y_%m%d_%H%M/")
	result_dir = "train_result/" + date_dir + f"{args.model}"
	if os.path.exists(result_dir) == False:
		os.makedirs(result_dir)
	model_save_path = result_dir + "/torch_model.pth"	# torch model's path
	log_save_path = result_dir + "/train_log.log"
	logger_ = create_logger(log_save_path)
	#logger_.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
	logger_.info('\n'.join('%s: %s' % (k, str(v)) for k, v in dict(vars(args)).items()))
	logger_.info("torch model and log file saved directory:")
	logger_.info(os.getcwd() + "train_result/" + date_dir)
	torch.manual_seed(args.torch_seed)
	torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	process_module = import_module("preprocess")
	torch.set_printoptions(precision = 5, threshold =2000, edgeitems = 5, linewidth = 90, sci_mode = False)
	if args.bert == True:
		preprocess_bert = process_module.processor_bert(args)
		np_train_contents, np_dev_contents, np_test_contents = preprocess_bert.get_data()
	else:
		preprocess_ = process_module.processor(args)
		np_train_contents, np_dev_contents, np_test_contents, np_embeddings_subset = preprocess_.get_data()

	train_contents, dev_contents, test_contents = np_train_contents.tolist(), np_dev_contents.tolist(), np_test_contents.tolist()
	args.seq_length = len(train_contents[0][0])	# make sure to assign the correct seq len
	train_iter = process_module.build_batch_iter(args, train_contents, torch_device)
	dev_iter = process_module.build_batch_iter(args, dev_contents, torch_device)
	test_iter = process_module.build_batch_iter(args, test_contents, torch_device)
	model = import_module("models." + args.model).model_net(args).to(torch_device)
	trainer_ins = trainer(model, args, train_iter, test_iter, dev_iter, model_save_path, torch_device)
	trainer_ins.train_()
	logger_.info("Final torch model and log file saved directory:")
	logger_.info(os.getcwd() + "train_result/" + date_dir)




