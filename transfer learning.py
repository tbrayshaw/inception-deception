import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
import time
from datetime import datetime
from tensorflow.contrib.slim.nets import inception
from scipy.misc import imresize
from collections import Counter
from PIL import Image
from models.research.slim.datasets import dataset_utils
slim = tf.contrib.slim

train = 0.70
validation = 0.05
test = 0.25

def split_data(X, y, train=train, test=test, validation=validation):
    X = np.array(X)
    X_raw_test = []
    X_raw_valid = []
    X_raw_train = []
    y = np.array(y)
    y_raw_test = []
    y_raw_valid = []
    y_raw_train = []

    random_indices = np.random.permutation(len(X))
    X = X[random_indices]
    y = y[random_indices]

    for image, label in zip(X, y):
        test_length = math.floor(test * class_images[label])
        valid_length = math.floor(validation * class_images[label])

        if Counter(y_raw_test)[label] < test_length:
            X_raw_test.append(image)
            y_raw_test.append(label)
        elif Counter(y_raw_valid)[label] < valid_length:
            X_raw_valid.append(image)
            y_raw_valid.append(label)
        else:
            X_raw_train.append(image)
            y_raw_train.append(label)

    return np.array(X_raw_train, dtype=np.float32), \
           np.array(X_raw_valid, dtype=np.float32), \
           np.array(X_raw_test, dtype=np.float32), \
           np.array(y_raw_train, dtype=np.int32), \
           np.array(y_raw_valid, dtype=np.int32), \
           np.array(y_raw_test, dtype=np.int32)

def plot_color_image(image):
    plt.figure(figsize=(4,4))
    plt.imshow(image.astype(np.uint8), interpolation='nearest')
    plt.axis('off')

def preprocess(image):
    image = imresize(image, (299, 299)) / 255
    return image

def batch(X, y, start_index=0, batch_size=4):
    stop_index = start_index + batch_size
    prepared_images = []
    labels = []

    for index in range(start_index, stop_index):
        prepared_images.append(preprocess(X[index]))
        labels.append(y[index])

    X_batch = np.stack(prepared_images)
    y_batch = np.array(labels, dtype=np.int32)

    return X_batch, y_batch

if not tf.gfile.Exists("models/cnn"):
    tf.gfile.MakeDirs("models/cnn")

if not os.path.exists("models/cnn/inception_v3.ckpt"):
    dataset_utils.download_and_uncompress_tarball("http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz", "models/cnn")

if not os.path.exists("tmp/faces"):
    os.makedirs("tmp/faces")

if not os.path.exists("tmp/faces/lfw-deepfunneled.tgz"):
    dataset_utils.download_and_uncompress_tarball("http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz", "tmp/faces")

class_mapping = {}
class_images = {}
dir = enumerate(os.listdir("images/faces/"))

for index, directory in dir:
    a = directory.split(" ")
    class_mapping[index] = a[0]
    path = os.path.join("images/faces/", a[0])
    class_images[index] = len([f for f in os.listdir(path)])
print(class_mapping)

image_arrays = []
image_labels = []
root_image_directory = "images/faces/"
for label, person in class_mapping.items():
    for directory in os.listdir(root_image_directory):
        if directory == person:
            image_directory = root_image_directory + directory
            break

    for image in os.listdir(image_directory):
        image = plt.imread(os.path.join(image_directory, image))
        image_arrays.append(image)
        image_labels.append(label)
image_arrays = np.array(image_arrays)
image_labels = np.array(image_labels)


X = np.array(image_arrays)
X_raw_test = []
X_raw_valid = []
X_raw_train = []
y = np.array(image_labels)
y_raw_test = []
y_raw_valid = []
y_raw_train = []

random_indices = np.random.permutation(len(X))
X = X[random_indices]
y = y[random_indices]

for image, label in zip(X, y):
    test_length = math.floor(test * class_images[label])
    valid_length = math.floor(validation * class_images[label])

    if Counter(y_raw_test)[label] < test_length:
        X_raw_test.append(image)
        y_raw_test.append(label)
    elif Counter(y_raw_valid)[label] < valid_length:
        X_raw_valid.append(image)
        y_raw_valid.append(label)
    else:
        X_raw_train.append(image)
        y_raw_train.append(label)

X_train = np.array(X_raw_train, dtype=np.float32)
X_valid = np.array(X_raw_valid, dtype=np.float32)
X_test = np.array(X_raw_test, dtype=np.float32)
y_train = np.array(y_raw_train, dtype=np.int32)
y_valid = np.array(y_raw_valid, dtype=np.int32)
y_test = np.array(y_raw_test, dtype=np.int32)

for class_number, person in class_mapping.items():
    print(f"{class_number}. {person}: {class_images[class_number]}")


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 299, 299, 3], name='X')
is_training = tf.placeholder_with_default(False, [])

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=is_training)

inception_saver = tf.train.Saver()

prelogits = tf.squeeze(end_points['PreLogits'], axis=[1, 2])
n_outputs = len(class_mapping)

people_logits = tf.layers.dense(prelogits, n_outputs, name="people_logits")
probability = tf.nn.softmax(people_logits, name='probability')

y = tf.placeholder(tf.int32, None)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=people_logits, labels=y))
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="people_logits")
training_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, var_list=train_vars)

correct = tf.nn.in_top_k(predictions=people_logits, targets=y, k=1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

X_valid, y_valid = batch(X_valid, y_valid, 0, len(X_valid))

with tf.name_scope("tensorboard"):
    valid_acc_summary = tf.summary.scalar(name='valid_acc', tensor=accuracy)
    valid_loss_summary = tf.summary.scalar(name='valid_loss', tensor=loss)
    train_acc_summary = tf.summary.scalar(name='train_acc', tensor=accuracy)
    valid_merged_summary = tf.summary.merge(inputs=[valid_acc_summary, valid_loss_summary])

    n_epochs = 100
    batch_size = 50
    max_checks_without_progress = 10
    checks_without_progress = 0
    best_loss = np.float("inf")
    show_progress = 1
    n_iterations_per_epoch = len(X_train) // batch_size
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{now}_unaugmented"
    logdir = "tensorboard/faces/" + model_dir
    file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
    inception_v3_checkpoint_path = "models/cnn/inception_v3.ckpt"
    unaugmented_training_path = "models/cnn/inception_v3_faces_unaugmented.ckpt"

with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, inception_v3_checkpoint_path)

    t0 = time.time()
    for epoch in range(n_epochs):
        start_index = 0

        for iteration in range(n_iterations_per_epoch):
            X_batch, y_batch = batch(X_train, y_train, start_index, batch_size)
            sess.run(training_op, {X: X_batch, y: y_batch})
            start_index += batch_size

        if epoch % show_progress == 0:
            train_summary = sess.run(train_acc_summary, {X: X_batch, y: y_batch})
            file_writer.add_summary(train_summary, (epoch + 1))
            valid_loss, valid_acc, valid_summary = sess.run([loss, accuracy, valid_merged_summary], {X: X_valid, y: y_valid})
            file_writer.add_summary(valid_summary, (epoch + 1))
            print(f"Epoch: {epoch+1} Loss: {valid_loss} Accuracy: {valid_acc}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            checks_without_progess = 0
            save_path = saver.save(sess, unaugmented_training_path)
        else:
            checks_without_progress += 1

        if checks_without_progress > max_checks_without_progress:
            print(f"Stopping Early. Loss has not improved in {max_checks_without_progress} epochs.")
            break

    t1 = time.time()

print('Training Time: {:.2f} minutes'.format((t1 - t0) / 60))

eval_batch_size = 32
n_iterations = len(X_test) // eval_batch_size
with tf.Session() as sess:
    saver.restore(sess, unaugmented_training_path)

    start_index = 0
    test_accuracy = {}

    t0 = time.time()
    for i in range(n_iterations):
        X_test_batch, y_test_batch = batch(X_test, y_test, start_index, batch_size=eval_batch_size)
        test_accuracy[i] = accuracy.eval({X: X_test_batch, y: y_test_batch})
        start_index += eval_batch_size
        print('Iteration: {} Batch Testing Accuracy: {:.2f}%'.format(i + 1, test_accuracy[i] * 100))

t1 = time.time()

print('\nFinal Testing Accuracy: {:.4f}% on {} instances.'.format(np.mean(list(test_accuracy.values())) * 100, len(X_test)))
print('Total evaluation time: {:.4f} seconds'.format((t1 - t0)))

def classify_image(index, images=X_test, labels=y_test):
    image_array = images[index]
    label = class_mapping[labels[index]]

    prepared_image = preprocess(image_array)
    prepared_image = np.reshape(prepared_image, newshape=(-1, 299, 299, 3))

    predictions = sess.run(probability, {X: prepared_image})

    predictions = [(i, prediction) for i, prediction in enumerate(predictions[0])]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    print('\nCorrect Answer: {}'.format(label))
    print('\nPredictions:')
    for prediction in predictions:
        class_label = prediction[0]
        probability_value = prediction[1]
        label = class_mapping[class_label]
        print("{:26}: {:.2f}%".format(label, probability_value * 100))

    plot_color_image(image_array)
    return predictions

with tf.Session() as sess:
    init.run()
    saver.restore(sess, "models/cnn/inception_v3_faces_unaugmented.ckpt")
    for i in range(10):
        classify_image(random.randint(1, 360))