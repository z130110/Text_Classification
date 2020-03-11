import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import random
from sklearn import manifold
from scipy.stats import zscore
#from models.TextCNN import textcnn_net
# from train import trainer_
random.seed(1)

class feat_reduction(object):
	def __init__(self, test_data = "", trained_model = "", tsne_2d = "", save_plot = ""):
		self.test_data = test_data
		self.trained_model = trained_model
		self.tsne_2d = tsne_2d
		self.save_plot = save_plot

	def load_np(self):
		np_data = np.load(self.test_data, allow_pickle = True)
		np_x = []
		np_y = []
		for i in range(np_data.shape[0]):
			np_x.append(np_data[i][0])
			np_y.append(np_data[i][1])
		np_x = np.array(np_x)
		np_y = np.array(np_y)
		self.np_y = np_y
		return np_x, np_y

	def extract_feat_vec(self):
		torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#model_path = "train_result/" + "2020_0217_2039(full_epoch)/torch_model.pth"
		np_x, np_y = self.load_np()
		tensor_x = torch.LongTensor(np_x).to(torch_device)
		model =torch.load(self.trained_model, map_location = torch_device)
		model.eval()
		feat_vec = model(tensor_x).detach().numpy()
		return feat_vec

	# def hook(module, input_data, out_put):
	#     '''把这层的输出拷贝到features中'''
	#     print(out_put.data)
	# #handle = model.conv_module.register_forward_hook(hook)

	def tsne_reduction(self):
		if os.path.exists(self.tsne_2d):
			tsne_transform_2d = np.load(self.tsne_2d)
			self.load_np()
		else:
			x = self.extract_feat_vec()
			standardize_feat = zscore(x, axis = 1)
			time_start = time.time()
			print("Start tsne reduction ...")
			tsne_transform_2d = manifold.TSNE(n_components = 2, random_state= 1).fit_transform(standardize_feat).tolist()
			print('t-SNE done! time elapsed: {} seconds'.format(time.time()-time_start))
			tsne_transform_2d = np.array(tsne_transform_2d)
			np.save(self.tsne_2d, tsne_transform_2d)
		return tsne_transform_2d	

	def plot_2d(self):
		x_2d = self.tsne_reduction()
		num_categories = len(set(self.np_y))
		css_colors_dict = mcolors.CSS4_COLORS
		cagegories = list(set(self.np_y))
		color_selection = ["rosybrown", "brown", "red", "orangered", "peru", "orange", "olive", "chartreuse", \
							"darkgreen", "turquoise", "teal", "deepskyblue", "dodgerblue", "navy", "blue", "slateblue",\
							"blueviolet", "violet", "purple", "magenta", "deeppink", "pink"]
		css_values = list(css_colors_dict.values())
		color_category = random.sample(color_selection, k = num_categories)
		categor_map_color = {cat:color for cat, color in zip(cagegories, color_category)}
		
		plot_dict = {}
		for i in range(x_2d.shape[0]):
			point = x_2d[i].tolist()
			label = self.np_y[i]
			if label in plot_dict.keys():
				plot_dict[label].append(point)
			else:
				plot_dict[label] = []
				plot_dict[label].append(point)
		
		fig, ax = plt.subplots(figsize=(6.5, 5))	
		for category in plot_dict.keys():
			x_points, y_points = np.array(plot_dict[category])[:,0], np.array(plot_dict[category])[:,1]
			color = categor_map_color[category]
			ax.scatter(x_points, y_points, s=7, c = color, label = str(category), alpha=0.4, edgecolors='none')

		ax.legend(borderpad = 0.3, labelspacing = 0.3, markerscale= 2, \
					loc='center right', bbox_to_anchor=(1.07, 0.55))
		#plt.rcParams['font.family'] = ['SimHei']
		ax.grid(linestyle='dotted')
		ax.axis('off')
		ax.axis('tight')
		ax.set_title(" Visualization of sentence features(TextCNN)", \
					fontdict={'fontsize': 9})
		plt.xlim(-100, 105)
		plt.ylim(-110, 120)
		if self.save_plot:
			plt.savefig(self.save_plot, dpi=600, bbox_inches='tight')
		plt.show()	


if __name__ == "__main__":
	test_data = "data/liquid/test_index.npy"
	save_dir = "feat_reduction/TextCNN_liquid/"
	trained_model = save_dir + "torch_model.pth"
	tsne_2d = save_dir + "tsne_liquid_textcnn.npy"
	save_plot = save_dir + "liquid_textcnn.pdf"
	reduct_obj = feat_reduction(test_data, trained_model, tsne_2d, save_plot)
	reduct_obj.plot_2d()
























