from __future__ import division
from __future__ import print_function

import argparse
import sys
import random
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

import os
import numpy as np
import tensorflow as tf

import dataset_utils
from retrieval_model import setup_train_model

feat_model = 'bertvit'
dataset = 'harmc'
ncls = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = None

def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def main(_):
    # Load data.
    print('Reading Img mod')
    train_file = './encoded_Labels/%s_img_%s_train.csv'%(feat_model, dataset)
    test_file = './encoded_Labels/%s_img_%s_test.csv'%(feat_model, dataset)

    data, labels = dataset_utils.read_csv(train_file)
    data_test, labels_test = dataset_utils.read_csv(test_file)
    
    print('Reading Txt mod')
    train_file_txt = './encoded_Labels/%s_txt_%s_train.csv'%(feat_model, dataset)
    test_file_txt = './encoded_Labels/%s_txt_%s_test.csv'%(feat_model, dataset)

    data_txt, labels_txt = dataset_utils.read_csv(train_file_txt)
    data_test_txt, labels_test_txt = dataset_utils.read_csv(test_file_txt)

    train_set = data
    train_label = labels
    test_set = data_test
    test_label = labels_test

    train_set_txt = data_txt
    train_label_txt = labels_txt
    test_set_txt = data_test_txt
    test_label_txt = labels_test_txt
    
    
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    test_label = le.transform(test_label)
    test_label_txt = le.transform(test_label_txt)
    
    if ncls == 2:
        print('making classes 2')
        train_label = np.asarray(train_label)
        test_label = np.asarray(test_label)
        train_label_txt = np.asarray(train_label_txt)
        test_label_txt = np.asarray(test_label_txt)
        
        train_label[np.where(train_label==2)] = 1
        test_label[np.where(test_label==2)] = 1
        train_label_txt[np.where(test_label_txt==2)] = 1
        test_label_txt[np.where(test_label_txt==2)] = 1
    
    # dataset statistics
    print("Train file length Img", len(train_set))
    print("Train file label length Img", len(train_label))
    print("Test file length Img", len(test_set))
    print("Test label length Img", len(test_label))

    print("Train file length txt", len(train_set_txt))
    print("Train file label length txt", len(train_label_txt))
    print("Test file length txt", len(test_set_txt))
    print("Test label length txt", len(test_label_txt))

    combined = list(zip(train_set, train_set_txt, train_label))
    random.shuffle(combined)
    train_set[:], train_set_txt[:], train_label[:] = zip(*combined)

    print(np.unique(train_label))
    print(np.unique(test_label))
    print(np.unique(test_label_txt))

    # mean_data_img_train = np.mean(mm_data, axis=0)

    steps_per_epoch = len(train_set) // FLAGS.batch_size

    # Setup placeholders for input variables.
    input_images = tf.placeholder(tf.float32, shape=(None, FLAGS.input_emb_size), name='input_images')
    input_text = tf.placeholder(tf.float32, shape=(None, FLAGS.input_emb_size), name='input_text')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    train_phase = tf.placeholder(tf.bool)

    # Setup training operation.
    FLAGS.n_classes = ncls
    logits, total_loss, accuracy, probs = setup_train_model(input_images, input_text, labels, FLAGS.n_classes,
                                                            train_phase)

    # Setup optimizer.
    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = 1e-4
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               steps_per_epoch, 0.794, staircase=True)
    optim = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optim.minimize(total_loss, global_step=global_step)

    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=FLAGS.epochs)    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        num_train_samples = len(train_set)
        num_test_samples = len(test_set)
        num_of_batches = num_train_samples // FLAGS.batch_size
        num_of_batches_test = num_test_samples // FLAGS.batch_size

        for epoch in range(FLAGS.epochs):
            test_acc = 0.
            multi_accuracy = 0.
            test_acc_txt = 0.
            train_acc = 0.
            train_loss = 0.
            test_loss = 0.
            test_loss_txt = 0.

            for idx in range(num_of_batches):
                batch_images, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, train_set)
                batch_images_text, _ = get_batch(idx, FLAGS.batch_size, train_label, train_set_txt)
                _, _, train_batch_loss, train_batch_acc, _ = sess.run(
                    [train_step, logits, total_loss, accuracy, probs],
                    feed_dict={
                        input_images: batch_images,
                        input_text: batch_images_text,
                        labels: batch_labels,
                        train_phase: True,
                    })

                train_acc += train_batch_acc
                train_loss += train_batch_loss

            train_acc /= num_of_batches
            train_acc = train_acc * 100
            train_loss /= num_of_batches

            for s_batch in range(num_of_batches_test):
                batch_images_test, batch_labels_test = get_batch(s_batch, FLAGS.batch_size, test_label, test_set)
                batch_images_test_text, _ = get_batch(s_batch, FLAGS.batch_size, test_label, test_set_txt)
                img_logits, test_loss_batch, test_batch_acc, probabilities_img = sess.run(
                    [logits, total_loss, accuracy, probs],
                    feed_dict={
                        input_images: batch_images_test,
                        input_text: batch_images_test_text,
                        labels: batch_labels_test,
                        train_phase: False,
                    })

                multi_accuracy += test_batch_acc
                test_loss += test_loss_batch

            multi_accuracy /= num_of_batches_test
            multi_accuracy = multi_accuracy * 100

            for s_batch in range(num_of_batches_test):
                batch_images_test, batch_labels_test = get_batch(s_batch, FLAGS.batch_size, test_label, test_set)
                batch_images_test_text, _ = get_batch(s_batch, FLAGS.batch_size, test_label, test_set_txt)
                img_logits, test_loss_batch, test_batch_acc, probabilities_img = sess.run(
                    [logits, total_loss, accuracy, probs],
                    feed_dict={
                        input_images: batch_images_test,
                        input_text: batch_images_test_text * 0.0,
                        labels: batch_labels_test,
                        train_phase: False,
                    })

                test_acc += test_batch_acc
                test_loss += test_loss_batch

            test_acc /= num_of_batches_test
            test_acc = test_acc * 100

            for s_batch in range(num_of_batches_test):
                batch_images_test, batch_labels_test = get_batch(s_batch, FLAGS.batch_size, test_label, test_set)
                batch_images_test_text, _ = get_batch(s_batch, FLAGS.batch_size, test_label, test_set_txt)
                img_logits, test_loss_batch, test_batch_acc, probabilities_img = sess.run(
                    [logits, total_loss, accuracy, probs],
                    feed_dict={
                        input_images: batch_images_test * 0.0,
                        input_text: batch_images_test_text,
                        labels: batch_labels_test,
                        train_phase: False,
                    })

                test_acc_txt += test_batch_acc
                test_loss_txt += test_loss_batch
            
            test_acc_txt /= num_of_batches_test
            test_acc_txt = test_acc_txt * 100

            print((
                "Epoch: {}, Train_Acc:{:.4f}, Train_Loss:{:.4f}, Multi_Acc_img:{:.4f}, Test_Acc_img:{:.4f}, Test_loss_img:{:.4f}, Test_Acc_txt:{:.4f}, Test_loss_txt:{:.4f}".
                format(epoch, train_acc, train_loss, multi_accuracy, test_acc, test_loss, test_acc_txt,
                       test_loss_txt)))
            
            save = '%s_%s_%d'%(feat_model, dataset, ncls)
            
            if not os.path.exists(save):
                os.makedirs(save)
            
            save_dir = '%s/ep%04d_mm_acc%0.3f/'%(save, epoch, multi_accuracy)
            saver.save(sess, save_dir, global_step=global_step)

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--input_emb_size', type=int, default=768, help='Input embedding size.')
    parser.add_argument('--n_classes', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Batch size for training.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
