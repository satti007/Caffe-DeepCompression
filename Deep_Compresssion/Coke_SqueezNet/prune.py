import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def prune(file,iter):
	
	f = np.load('Wts/'+file)
	initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	
	f = np.load('mask.npz')
	old_mask = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	
	sess = tf.Session()
	q_param = 0.0001
	new_mask = []
	threshold = []
	params_count = []
	pruned_weights = []
	pruned_params_count = []
	for i in range(7,17):
		params_count.append(np.count_nonzero(initial_weights[i]))
		threshold.append(sess.run(tf.sqrt(tf.nn.l2_loss(initial_weights[i])) * q_param))
		new_mask.append((initial_weights[i] > threshold[i-7]).astype(int))
		new_mask[i-7] = np.multiply(new_mask[i-7],old_mask[i-7])
		pruned_weights.append(np.multiply(initial_weights[i],new_mask[i-7]))
		initial_weights[i]=pruned_weights[i-7]
		pruned_params_count.append(np.count_nonzero(pruned_weights[i-7]))
	
	print params_count
	print pruned_params_count
	print sum(pruned_params_count)
	
	mask = np.asarray(new_mask)
	np.savez('mask.npz', *mask)
	np.savez('Wts/pruned_weights_'+str(iter)+'.npz', *initial_weights)


prune('PR_Squeeze_Weights_5.npz',6)
# f = np.load('Wts/Latest_Sq_Wts_100_100_18_Apr.npz')
A=[]
for i in range (1,6):
f = np.load('Wts/pruned_weights_5.npz')
initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
params_count=[]
for j in range(7,17):
	params_count.append(np.count_nonzero(initial_weights[j]))

sum(params_count)