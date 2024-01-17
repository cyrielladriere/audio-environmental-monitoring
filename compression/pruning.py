# https://github.com/Arshdeep-Singh-Boparai/Efficient_CNNs_passive_filter_pruning/blob/main/Proposed_pruning.py
# Script to obtain importance of filters using the proposed operato norm pruning method, entry-wise l_1 norm based scores and geometric median based scores.
import numpy as np    
import os
import torch
from scipy.stats.mstats import gmean
from scipy.spatial import distance

model_pann_trained = "resources/model_pann.pt"

# os.chdir('~/importance_scores/VGG16_MNIST/')

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
   
# load weights from the unpruned network (we have used numpy format to save  and load the pre-trained weights)

# W_init = list(np.load('/~/VGG_MNIST/VGG_MNIST_baseline_200/best_weights_numpy.npy', allow_pickle=True))#list(np.load('/home/arshdeep/Pruning/SPL/VGG_pruned_Model/VGG-CIFAR100_Pruning/data/VGG_weights100.npy',allow_pickle=True))
W_init = torch.load(model_pann_trained)
W_init = [tensor.cpu().numpy() for tensor in W_init.values()]  # only need the weights, no need for keys because we have the indexes 

# Obtaining layer-wise importance scores of CNN filters
indexes = [8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158, 164, 170, 176, 182, 188, 194, 200, 206, 212, 218, 224, 230, 236, 242, 248, 254, 260, 266, 272, 278, 284, 290, 296, 302, 308, 314]# indexes=[0,6,12,18,24,30,36,42,48,54,60,66,72] # indexes of convolution layers in W_init for VGG-16
# L=[1,2,3,4,5,6,7,8,9,10,11,12,13]  # convolutional layer number (for VGG-16, it is from 1 to 13)


for i, j in enumerate(indexes):
	# print(i)
	W_2D=W_init[j]
	# print(W_2D.shape)
	W=np.reshape(W_2D,(np.shape(W_2D)[2]*np.shape(W_2D)[3],np.shape(W_2D)[1],np.shape(W_2D)[0]))  # (kernel_size x kernel_size, channels_in_filter, amount_of_filters) different arrangement from paper
	# print(np.shape(W),'layer  :','  ', j)
	# print(np.shape(W),'shape of weights')
	score_norm_m1 = operator_norm_pruning(W)
	# print(np.argsort(score_norm_m1))
	# Score_L1=CVPR_L1_Imp_index(W)  #l_1 entry wise norm based important scores
	# Score_GM=CVPR_GM_Imp_index(W)  #Geomettric median based important scores
	file_name='opnorm_pruning_layer_'+str(i)+'.npy'
	np.save(f"compression/pruning_scores/{file_name}",np.argsort(score_norm_m1)) # save sorted arguments from low to high importance.
	