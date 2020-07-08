import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
import json
import PIL
from random import randint

# Change this URL to change the input image. Should be an image of something in the ImageNet library (imagenet_labels.json)
imageUrl = "http://fergusontalon.com/wp-content/uploads/2018/12/image1-9.jpg"

numTopProbs = 5
target_class = randint(0, 999)

def inception(image, reuse):
    img_expanded = tf.expand_dims(image, 0)
    img_subtracted = tf.subtract(img_expanded, 0.5)
    img_preprocessed = tf.multiply(img_subtracted, 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(img_preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:]
        probs = tf.nn.softmax(logits)
    return logits, probs

def classify(img, title):
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(11, 5))
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.title.set_text(title)
    ax1.imshow(img)
    ax1.axis("off")

    topk = list(p.argsort()[-numTopProbs:][::-1])
    topprobs = p[topk]

    barlist = ax2.barh(range(numTopProbs), topprobs)
    ax2.set_yticks(range(numTopProbs))
    ax2.set_yticklabels([imagenet_labels[i][:15] for i in topk])
    ax2.invert_yaxis()
    ax2.set_xlabel("Classification Confidence")

    labels = []
    for bar in barlist:
        width = bar.get_width()
        if (width < 0.5):
            x = width + 0.02
            weight = 'normal'
            colour = 'grey'
            align = 'left'
        else:
            x = width * 0.98
            weight = 'bold'
            colour = 'white'
            align = 'right'

        y = bar.get_y() + (bar.get_height() / 2)
        label = ax2.text(x, y, "{0:.5%}".format(width), horizontalalignment=align,
                         verticalalignment='center', color=colour, weight=weight, clip_on=True)
        labels.append(label)

    print(f"Classification: {imagenet_labels[topk[0]]}")
    print("Confidence: {0:.10%}\n".format(topprobs[0]))

    plt.show()

def adv_plot_noise(diff):
    min = diff.min()
    max = diff.max()

    norm = (diff - min) / (max - min)

    plt.imshow(norm)
    plt.title("Adversarial Noise")
    plt.axis("off")
    plt.show()


labels_resource = 'imagenet_labels.json'
inceptionUrl = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'

data_dir = 'models/cnn'
checkpointPath = 'models/cnn/inception_v3.ckpt'

print("Creating Tensorflow session...")
sess = tf.InteractiveSession()
print("Tensorflow session created.\n")
image = tf.Variable(tf.zeros((299,299,3)))

print("Retrieving logits and probabilities from Inception model...")
logits, probs = inception(image, reuse=False)
print("Logits and probabilities retrieved.\n")

if (os.path.exists(checkpointPath) == False):
    print(f"Inception v3 checkpoint does not exist in path: {checkpointPath}")
    print(f"Retrieving Inception v3 from URL: {inceptionUrl}")
    inception_tarball, _ = urlretrieve(inceptionUrl)
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)
    print(f"Inception model extracted to path: {data_dir}\n")

print("Restoring Inception variables...")
vars = [var for var in tf.global_variables() if var.name.startswith('InceptionV3/')]
print("Inception variables restored.\n")
saver = tf.train.Saver(vars)
print("Restoring Inception checkpoint...")
saver.restore(sess, checkpointPath)
print("Inception checkpoint restored.\n")

print(f"Loading ImageNet labels from resource: {labels_resource}")
with open(labels_resource) as f:
    imagenet_labels = json.load(f)
print("ImageNet labels loaded.\n")

print(f"Loading input image from URL: {imageUrl}")

img_path, _ = urlretrieve(imageUrl)
print("Image loaded.")

img = PIL.Image.open(img_path)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
print(f"Image is {img.width}x{img.height}... resizing to {new_w}x{new_h}.\n")
img = img.resize((new_w, new_h))
img = img.crop((np.floor(new_w/2 - 149), 0, np.floor(new_w/2 + 150), 299)) \
    if wide else img.crop((0, np.floor(new_h/2 - 149), 299, np.floor(new_h/2 + 150)))
img = (np.asarray(img) / 255.0).astype(np.float32)

print("Classifying image...")
classify(img, "Original Scaled Input Image")


print("Starting Adversarial process...\n")
lr = 0.1
epsilon = 8e-3
steps = 50
print(f"Learning Rate: {lr}")
print(f"Epsilon: {epsilon}")
print(f"Maximum Steps: {steps}")
print(f"Target Class: {target_class} - {imagenet_labels[target_class]}\n")




input_placeholder = tf.placeholder(tf.float32, (299, 299, 3))
input_image = image
assign_input_to_placeholder = tf.assign(input_image, input_placeholder)

learning_rate_placeholder = tf.placeholder(tf.float32, ())
classification_label_placeholder = tf.placeholder(tf.int32, ())

labels = tf.one_hot(classification_label_placeholder, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(learning_rate_placeholder).minimize(loss, var_list=[input_image])

epsilon_ph = tf.placeholder(tf.float32, ())
below = input_placeholder - epsilon_ph
above = input_placeholder + epsilon_ph

projected = tf.clip_by_value(tf.clip_by_value(input_image, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(input_image, projected)

sess.run(assign_input_to_placeholder, feed_dict={input_placeholder: img})

last_loss = 10

for i in range(steps):
    if(last_loss > 0.01):
        _, loss_value = sess.run(
            [optim_step, loss],
            feed_dict={learning_rate_placeholder: lr,
                       classification_label_placeholder: target_class})

        last_loss = loss_value

        grad = sess.run(project_step, feed_dict={input_placeholder: img, epsilon_ph: epsilon})

        print(f"Step {i+1}, Loss = {loss_value}")
    else:
        print(f"Stopping early - minimal loss (classification confidence above 99%) reached at step {i}\n")
        break

adv = input_image.eval()
diff = np.abs(np.subtract(img, adv))
adv_plot_noise(diff)

print("\nClassifying perturbed image with adversarial noise...")
print(f"ADVERSARIAL ATTACK HAS SUCCEEDED IF CLASSIFICATION IS: {imagenet_labels[target_class]}")
classify(adv, "Original Image + Adversarial Noise")





ex_angle = np.pi/8

print("Rotating input image...")
rotation = tf.placeholder(tf.float32, ())
rotated_image = tf.contrib.image.rotate(image, rotation)
rotated_example = rotated_image.eval(feed_dict={image: adv, rotation: ex_angle})
print("Classifying rotated input image...")
classify(rotated_example, "Rotated Adversarial Image")

num_samples = 30
average_loss = 0

print("Training model on rotated input images...")
for i in range(num_samples):
    rotated = tf.contrib.image.rotate(image, tf.random_uniform((), minval=-np.pi/4, maxval=np.pi/4))
    rotated_logits, _ = inception(rotated, reuse=True)
    average_loss += tf.nn.softmax_cross_entropy_with_logits_v2(logits=rotated_logits, labels=labels) / num_samples
    print(f"Rotated sample {i+1}/{num_samples} trained.")

optim_step = tf.train.GradientDescentOptimizer(learning_rate_placeholder).minimize(average_loss, var_list=[input_image])
print("Model trained and optimisation step updated.\n")




epsilon = 3e-2
lr = 0.2
steps = 50

print("Starting adversarial process for rotated samples...")

sess.run(assign_input_to_placeholder, feed_dict={input_placeholder: img})

last_loss = 10
for i in range(steps):
    if (last_loss > 0.01):
        _, loss_value = sess.run(
            [optim_step, average_loss],
            feed_dict={learning_rate_placeholder: lr, classification_label_placeholder: target_class})
        sess.run(project_step, feed_dict={input_placeholder: img, epsilon_ph: epsilon})

        last_loss = loss_value
        print(f"Step {i + 1}, Loss = {loss_value}")
    else:
        print(f"Stopping early - minimal loss (classification confidence above 99%) reached at step {i}")
        break

adv_robust = input_image.eval()

print("\nClassifying rotated image with adversarial noise...")
print(f"ADVERSARIAL ATTACK HAS SUCCEEDED IF CLASSIFICATION IS: {imagenet_labels[target_class]}")
rotated_example = rotated_image.eval(feed_dict={image: adv_robust, rotation: ex_angle})
classify(rotated_example, "Rotated Original Image + Adversarial Noise")


print("\nBeginning robustness evaluation...")
angles = np.linspace(-np.pi / 4, np.pi / 4, 301)

probs_naive = []
probs_robust = []
for angle in angles:
    print(f"Completing rotation of {angle}")
    rotated = rotated_image.eval(feed_dict={image: adv_robust, rotation: angle})
    probs_robust.append(probs.eval(feed_dict={image: rotated})[0][target_class])

    rotated = rotated_image.eval(feed_dict={image: adv, rotation: angle})
    probs_naive.append(probs.eval(feed_dict={image: rotated})[0][target_class])

print("\nPlotting robustness evaluation...")
print("Plot should show that the robust adversarial model performs well with rotation invariance.")
robust_line, = plt.plot(angles, probs_robust, color='g', linewidth=2, label='Robust Model')
naive_line, = plt.plot(angles, probs_naive, color='b', linewidth=2, label='Naive Model')
plt.ylim([0, 1.05])
plt.xlabel('Angle of Rotation')
plt.ylabel('Probability of Adversarial Success')
plt.legend(handles=[robust_line, naive_line], loc='lower right')
plt.show()