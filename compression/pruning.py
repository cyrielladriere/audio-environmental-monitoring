# https://github.com/Arshdeep-Singh-Boparai/Efficient_CNNs_passive_filter_pruning/blob/main/Proposed_pruning.py
# Script to obtain importance of filters using the proposed operato norm pruning method, entry-wise l_1 norm based scores and geometric median based scores.
import numpy as np    
import torch
from scipy.stats.mstats import gmean
from torch import nn, optim
from compression.training import train_model

def pruned_fine_tuning(model_pruned, P, model_dir, dataloaders, n_epochs, data, threshold, batch_size, TENSORBOARD, writer=None, opnorm=True):
	lr = 0.003 if opnorm else 0.005
	
	optimizer = optim.Adam(model_pruned.parameters(), lr=lr) 
	exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
	model_pruned.cuda()
	if TENSORBOARD:
		model_pruned = train_model(model_pruned, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD, writer)
		writer.flush()
		writer.close()

		filename = f"{model_dir}/model_opnorm_pruning_{P}_FT.pt" if opnorm else f"{model_dir}/model_L1_norm_pruning_{P}_FT.pt"
		torch.save(model_pruned.state_dict(), filename)
	else:
		model_pruned = train_model(model_pruned, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD)

def import_pruned_weights(model_original, model_pruned, P, opnorm=True, cnn_14=False):

	if not cnn_14:
		max_layers = 51
		start_layer = 5
		end_layer = 161
	else:
		max_layers = 11
		start_layer = 5
		end_layer = 41

	with torch.no_grad():
		k = 0
		prev_pruned_weights = None  # in_channel
		for i, layer in enumerate(model_original.named_parameters()): 
			layer = layer[0]
			layer_name = layer.split(".")

			model_copy_pruned = model_pruned
			model_copy_original = model_original
			current_layer_pruned = None
			current_layer_original = None

			for j in range(len(layer_name)-1):
				current_layer_pruned = getattr(model_copy_pruned, layer_name[j])
				current_layer_original = getattr(model_copy_original, layer_name[j])
				model_copy_pruned = current_layer_pruned
				model_copy_original = current_layer_original
			
			if k <= max_layers: # just for the last batchnorm/activation layer so we dont get file doesnt exist error (todo better fix)
				if opnorm:
					pruned_weights = np.load(f"compression/pruning_scores/cnn_14/opnorm/opnorm_pruning_layer_{k}.npy")
				else:
					pruned_weights = np.load(f"compression/pruning_scores/cnn_14/L1_norm/L1_norm_pruning_layer_{k}.npy")
				pruned_weights = sorted(pruned_weights[int(np.ceil(len(pruned_weights)*P)):])   # out_channel

			if i >= start_layer and i < end_layer: # until last non-fully connected node
				# print(f"{layer}, {i} -----------------------------")
				W = current_layer_original.state_dict()
				for key in W.keys():
					if key == 'num_batches_tracked':
						continue
					if len(W[key].shape) == 1:   # Batchnorm or activation layer
						W[key] = W[key][prev_pruned_weights]
					else: # Convulutional layer
						if(current_layer_original.state_dict()[key].shape[1] == 1): # Conv layer with groups element
							W[key] = W[key][pruned_weights,:,:,:]
						else:
							W[key] = W[key][:,prev_pruned_weights,:,:]
							W[key] = W[key][pruned_weights,:,:,:]
						k += 1
						prev_pruned_weights = pruned_weights
					# print(W[key].shape, current_layer_original.state_dict()[key].shape)
				current_layer_pruned.load_state_dict(W)
				# Randomly initialize fully connected layers
		return model_pruned

def save_pruned_layers(model_pann_trained, opnorm=True, cnn_14=False):
	# W_init = list(np.load('/~/VGG_MNIST/VGG_MNIST_baseline_200/best_weights_numpy.npy', allow_pickle=True))#list(np.load('/home/arshdeep/Pruning/SPL/VGG_pruned_Model/VGG-CIFAR100_Pruning/data/VGG_weights100.npy',allow_pickle=True))
	W_init = torch.load(model_pann_trained)
	W_init = [tensor.cpu().numpy() for tensor in W_init.values()]  # only need the weights, no need for keys because we have the indexes 

	# Obtaining layer-wise importance scores of CNN filters
	if not cnn_14:
		indexes = [8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158, 164, 170, 176, 182, 188, 194, 200, 206, 212, 218, 224, 230, 236, 242, 248, 254, 260, 266, 272, 278, 284, 290, 296, 302, 308, 314]# indexes=[0,6,12,18,24,30,36,42,48,54,60,66,72] # indexes of convolution layers in W_init for VGG-16
	else:
		indexes = [8, 9, 20, 21, 32, 33, 44, 45, 56, 57, 68, 69]

	# L=[1,2,3,4,5,6,7,8,9,10,11,12,13]  # convolutional layer number (for VGG-16, it is from 1 to 13)


	for i, j in enumerate(indexes):
		# print(i)
		W_2D=W_init[j]
		# print(W_2D.shape)
		W=np.reshape(W_2D,(np.shape(W_2D)[2]*np.shape(W_2D)[3],np.shape(W_2D)[1],np.shape(W_2D)[0]))  # (kernel_size x kernel_size, channels_in_filter, amount_of_filters) different arrangement from paper
		# print(np.shape(W),'layer  :','  ', j)
		# print(np.shape(W),'shape of weights')
		if opnorm:
			score_norm_m1 = operator_norm_pruning(W)
			file_name = 'opnorm_pruning_layer_'+str(i)+'.npy'
			np.save(f"compression/pruning_scores/cnn_14/opnorm/{file_name}",np.argsort(score_norm_m1)) # save sorted arguments from low to high importance.
		else:
			score_L1= L1_Imp_index(W)  #l_1 entry wise norm based important scores
			file_name = 'L1_norm_pruning_layer_'+str(i)+'.npy'
			np.save(f"compression/pruning_scores/cnn_14/L1_norm/{file_name}",np.argsort(score_L1)) # save sorted arguments from low to high importance.

		# print(np.argsort(score_norm_m1))
		# Score_GM=CVPR_GM_Imp_index(W)  #Geomettric median based important scores

# Proposed pruning framework
def operator_norm_pruning(W):
	C_M=[]	
	mean_vec=[]
	# Take c'th channel of all filters
	for i in range(np.shape(W)[1]):
		A=W[:,i,:].T
		A_mean=np.mean(A,0)
		e=np.tile(A_mean,(np.shape(A)[0],1))
		A_centred=A-e
		mean_vec.append(A_mean)
		u,q,v=np.linalg.svd(A_centred)
		u1=np.reshape(u[:,0],(np.shape(A)[0],1))
		v1=np.reshape(v[0,:],(np.shape(A)[1],1))
		c_1=np.matmul(u1,v1.T)
		c_1_norm=c_1[0,:]/np.linalg.norm(c_1[0,:])
		C_M.append(c_1_norm)
	Score=[]
	# Take i'th filter
	for i in range(np.shape(W)[2]):
		Score.append(np.trace((np.matmul((W[:,:,i]-np.array(mean_vec).T).T,np.array(C_M).T))))
	Mse_score=(np.array(Score))**2
	Mse_score_norm=Mse_score/np.max(Mse_score)
	return Mse_score_norm

# entry-wise l_1 norm based scores
def L1_Imp_index(W):
	Score=[]
	for i in range(np.shape(W)[2]):
		Score.append(np.sum(np.abs(W[:,:,i])))
	return Score/np.max(Score)

# Geometric median based scores
def GM_Imp_index(W):
	G_GM=gmean(np.abs(W.flatten()))
	Diff=[]
	for i in range(np.shape(W)[2]):
		F_GM=gmean(np.abs(W[:,:,i]).flatten())
		Diff.append((G_GM-F_GM)**2)
	return Diff/np.max(Diff)	