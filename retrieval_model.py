import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected

CENTER_LOSS_ALPHA = 0.5

def add_fc(inputs, outdim, train_phase, scope_in):
    fc = fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                                             training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def opl_loss(features, labels):
    # Normalize the features
    features = tf.nn.l2_normalize(features, axis=1)
    
    # Reshape labels to match the shape required for comparison
    labels = tf.reshape(labels, [-1])
    
    # Create a mask where pairs of labels are the same
    labels_expanded = tf.expand_dims(labels, 0)
    mask = tf.equal(labels_expanded, tf.transpose(labels_expanded))
    mask = tf.cast(mask, tf.float32)
    
    # Mask out the diagonal (i.e., the same examples)
    eye = tf.eye(tf.shape(mask)[0], dtype=tf.float32)
    mask_pos = mask - eye
    mask_neg = 1 - mask
    
    # Compute the dot products
    dot_prod = tf.matmul(features, tf.transpose(features))
    
    # Compute positive and negative pair means
    pos_pairs_mean = tf.reduce_sum(mask_pos * dot_prod) / (tf.reduce_sum(mask_pos) + 1e-6)
    neg_pairs_mean = tf.reduce_sum(tf.abs(mask_neg * dot_prod)) / (tf.reduce_sum(mask_neg) + 1e-6)
    
    # Compute the loss
    loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)
    
    return loss, pos_pairs_mean, neg_pairs_mean

def git_loss(features, labels, num_classes):
    len_features = features.get_shape()[1]
    w_init = tf.contrib.layers.xavier_initializer()
    centers = tf.Variable(initial_value=w_init(shape=(int(num_classes), int(len_features)), dtype=tf.float32),
                          trainable=False)

    # centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
    #                           initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.square(features - centers_batch))

    # Pairwise differences
    diffs = (features[:, tf.newaxis] - centers_batch[tf.newaxis, :])
    diffs_shape = tf.shape(diffs)

    # Mask diagonal (where i == j)
    mask = 1 - tf.eye(diffs_shape[0], diffs_shape[1], dtype=diffs.dtype)
    diffs = diffs * mask[:, :, tf.newaxis]

    # combinaton of two losses
    loss2 = tf.reduce_mean(tf.divide(1, 1 + tf.square(diffs)))

    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = tf.divide(diff, tf.cast((1 + appear_times), tf.float32))
    diff = CENTER_LOSS_ALPHA * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)  # diff is used to get updated centers.
    # combo_loss = value_factor * loss + new_factor * loss2
    combo_loss = 1 * loss + 1 * loss2

    return combo_loss, centers_update_op

def attention_fun(Q, K, scaled_=True, masked_=False):
    attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

    if masked_:
        raise NotImplementedError

    attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
    return attention

def embedding_loss(feats, tfeats, labels, num_classes):
    clogits = tf.add(feats, tfeats)

    logits = fully_connected(clogits, num_classes, activation_fn=None,
                             scope='tmp')
    
    with tf.variable_scope('loss') as scope:
        with tf.name_scope('soft_loss'):
            softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            probs = tf.nn.softmax(logits)
            
        # scope.reuse_variables()
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    with tf.name_scope('git_loss'):
        git, _ = git_loss(clogits, labels, num_classes)
    with tf.name_scope('opl_loss'):
        opl, _, _ = opl_loss(clogits, labels)
    with tf.name_scope('loss/'):
        loss = softmax
        tf.summary.scalar('TotalLoss', loss)
    
    predictions = tf.argmax(probs, axis=1)
    
    TP = tf.count_nonzero(predictions * labels, dtype=tf.float32)  # True Positives
    FP = tf.count_nonzero(predictions * (1 - labels), dtype=tf.float32)  # False Positives
    FN = tf.count_nonzero((1 - predictions) * labels, dtype=tf.float32)  # False Negatives
    
    precision = TP / (TP + FP + tf.keras.backend.epsilon())  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + tf.keras.backend.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    
    return logits, loss, accuracy, predictions# returns total loss

def embedding_model(im_feats, text_feats, train_phase, fc_dim=768, embed_dim=768):
    
    # image branch
    im_feats = tf.nn.l2_normalize(im_feats, 1, epsilon=1e-10)
    im_fc1 = add_fc(im_feats, fc_dim, train_phase, 'im_embed_1')
    im_fc2 = fully_connected(im_fc1, embed_dim, activation_fn=None,
                                scope='im_embed_2')
    i_embed = tf.nn.l2_normalize(im_fc2, 1, epsilon=1e-10)

    # text branch
    text_feats = tf.nn.l2_normalize(text_feats, 1, epsilon=1e-10)
    txt_fc1 = add_fc(text_feats, fc_dim, train_phase, 'txt_embed_1')
    txt_fc2 = fully_connected(txt_fc1, embed_dim, activation_fn=None,
                                scope='txt_embed_2')
    txt_embed = tf.nn.l2_normalize(txt_fc2, 1, epsilon=1e-10)



    return i_embed,txt_embed

def setup_train_model(input_images, input_text, labels, num_classes, train_phase):
    i_embed, t_embed = embedding_model(input_images, input_text, train_phase=train_phase)
    logits, loss, accuracy, probs = embedding_loss(i_embed, t_embed, labels, num_classes)
    return logits, loss, accuracy, probs
