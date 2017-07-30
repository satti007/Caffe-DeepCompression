import math
import numpy as np

# To calucalte new drop_out probs(formula according to the deep compression paper)
def calculate_new_keep_prob(original_keep_prob, original_connections, retraining_connections):
    return 1.0 - ((1.0 - original_keep_prob) * math.sqrt(retraining_connections / original_connections))

def prune(iter,kp1,kp2):

    sess = tf.Session()
    print '######### Start ##########' 
    
    if iter==1:
        f = np.load("Uncompressed_weights.npz")
    else:
        f = np.load("Wts/Weights_CR_"+str(iter-1)+".npz")
    initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
    
    fc1_w = tf.Variable(initial_weights[4], trainable=False)
    fc2_w = tf.Variable(initial_weights[6], trainable=False)
    fc3_w = tf.Variable(initial_weights[8], trainable=False)
    sess.run(tf.global_variables_initializer())
    
    # Count the parametrs in each layer != 0
    nonzero_indicator1 = tf.to_float(tf.not_equal(fc1_w, tf.zeros_like(fc1_w)))
    nonzero_indicator2 = tf.to_float(tf.not_equal(fc2_w, tf.zeros_like(fc2_w)))
    nonzero_indicator3 = tf.to_float(tf.not_equal(fc3_w, tf.zeros_like(fc3_w)))

    count_parameters1 = tf.reduce_sum(nonzero_indicator1)
    count_parameters2 = tf.reduce_sum(nonzero_indicator2)
    count_parameters3 = tf.reduce_sum(nonzero_indicator3)

    print("Before pruning")
    print("W Parameter Counts:")
    print("FC1_Parameters: {0}".format(sess.run(count_parameters1)))
    print("FC2_Parameters: {0}".format(sess.run(count_parameters2)))
    print("FC3_Parameters: {0}".format(sess.run(count_parameters3)))
    
    # Store the no.of.parametrs in FC1,FC2 layer != 0  before pruning to update drop_out ratios
    c1 = sess.run(count_parameters1)
    c2 = sess.run(count_parameters2)
    
    # Mask -- set parameter value-->0 if it's < threshold 
    #prune_maskx -- pruning mask for Fcx layer
    #fcx_pruned -- pruned weights for Fcx layer
    prune_mask1 = tf.Variable(tf.ones_like(fc1_w), trainable=False, name='prune_mask1')
    fc1_pruned = tf.multiply(fc1_w, prune_mask1) 
    prune_mask2 = tf.Variable(tf.ones_like(fc2_w), trainable=False, name='prune_mask2')
    fc2_pruned = tf.multiply(fc2_w, prune_mask2)
    prune_mask3 = tf.Variable(tf.ones_like(fc3_w), trainable=False, name='prune_mask3')
    fc3_pruned = tf.multiply(fc3_w, prune_mask3)

    # Get the pruning threshold
    threshold = 0.0001
    t1 = tf.sqrt(tf.nn.l2_loss(fc1_w)) * threshold
    t2 = tf.sqrt(tf.nn.l2_loss(fc2_w)) * threshold
    t3 = tf.sqrt(tf.nn.l2_loss(fc3_w)) * threshold

    # For each layer get the index of parameter who value >= threshold
    indicator_matrix1 = tf.multiply(tf.to_float(tf.greater_equal(fc1_w, tf.ones_like(fc1_w) * t1)), prune_mask1)
    indicator_matrix2 = tf.multiply(tf.to_float(tf.greater_equal(fc2_w, tf.ones_like(fc2_w) * t2)), prune_mask2)
    indicator_matrix3 = tf.multiply(tf.to_float(tf.greater_equal(fc3_w, tf.ones_like(fc3_w) * t3)), prune_mask3)

    # Update the mask accodingly(to above step)
    update_mask1 = tf.assign(prune_mask1, indicator_matrix1)
    update_mask2 = tf.assign(prune_mask2, indicator_matrix2)
    update_mask3 = tf.assign(prune_mask3, indicator_matrix3)
    update_all_masks = tf.group(update_mask1, update_mask2, update_mask3)

    # Update the weights 
    prune_fc1 = fc1_w.assign(fc1_pruned)
    prune_fc2 = fc2_w.assign(fc2_pruned)
    prune_fc3 = fc3_w.assign(fc3_pruned)
    prune_all = tf.group(prune_fc1, prune_fc2, prune_fc3)

    sess.run(tf.global_variables_initializer())
    sess.run(update_all_masks)
    sess.run(prune_all)



    initial_weights[4] = sess.run(fc1_w)
    initial_weights[6] = sess.run(fc2_w)
    initial_weights[8] = sess.run(fc3_w)
    np.savez('Wts/Weights_C_'+str(iter)+'.npz', *initial_weights)

    print("After pruning")
    print("W Parameter Counts:")
    print("FC1_Parameters: {0}".format(sess.run(count_parameters1)))
    print("FC2_Parameters: {0}".format(sess.run(count_parameters2)))
    print("FC3_Parameters: {0}".format(sess.run(count_parameters3)))
    print('######### Done ##########')
    
    # Store the no.of.parametrs in FC1,FC2 layer != 0  after pruning to update drop_out ratios
    c1_retrain = sess.run(count_parameters1)
    c2_retrain = sess.run(count_parameters2)
    # Updating the drop_out ratios
    kp1 = calculate_new_keep_prob(kp1, c1, c1_retrain)
    kp2 = calculate_new_keep_prob(kp2, c2, c2_retrain)

    return kp1,kp2
