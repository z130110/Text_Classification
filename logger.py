import logging
import time
from datetime import timedelta

class LogFormatter():
	def __init__(self):
		self.start_time = time.time()
	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)
		year = time.strftime("%Y")
		month = time.strftime("%m")
		day = time.strftime("%d")
		hour = time.strftime("%H")
		minute = time.strftime("%M")
		prefix = f"{record.levelname} - {year}/{month}/{day} {hour}:{minute} -- {timedelta(seconds=elapsed_seconds)}"
		message = record.getMessage()
		message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
		return "%s - %s" % (prefix, message)

def create_logger(file_path, verbose = "info"):
	logger = logging.getLogger("my logger") 
	logger.setLevel(logging.INFO) 

	# 2、创建一个handler，用于写入日志文件 
	file_write = logging.FileHandler(file_path, mode = "w")
	file_write.setLevel(logging.DEBUG) 

	# 再创建一个handler，用于输出到控制台 
	console_level = logging.INFO if verbose == "info" else logging.DEBUG
	console = logging.StreamHandler() 
	console.setLevel(console_level) 

	# 3、定义handler的输出格式（formatter）
	#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
	log_formatter = LogFormatter()

	# 4、给handler添加formatter
	file_write.setFormatter(log_formatter) 
	console.setFormatter(log_formatter) 

	# 5、给logger添加handler 
	logger.addHandler(file_write) 
	logger.addHandler(console)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()
	logger.reset_time = reset_time()
	return logger

