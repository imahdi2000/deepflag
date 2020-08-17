import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

#===================== constants =====================
TRAIN_DATA_REPETITION = 10
PIC_X = 28
PIC_Y = 28

#===================== import images =====================
images = [f for f in glob.glob(r'C:\Users\Hermano\Desktop\flags\*\*')]
##print(files[0])
##print(os.path.basename(os.path.dirname(images[0])))
##for img in images:
##    print(os.path.basename(os.path.dirname(img)))

label_map = {'canada':0, 'china':1, 'japan':2, 'mexico':3, 'united_states':4}
label = [label_map[os.path.basename(os.path.dirname(img))] for img in images]
##print(label)

def read_image(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [PIC_X, PIC_Y])
    return image_resized, label

dataset = tf.data.Dataset.from_tensor_slices((images, label))
dataset = dataset.map(read_image).shuffle(100).batch(10, drop_remainder = True)
#dataset = dataset.map(read_image).shuffle(100)

##print(dataset[0])
##print(tf.compat.v1.data.get_output_shapes(dataset))
##print(tf.compat.v1.data.get_output_types(dataset))

##for elem in dataset:
##    plt.figure()
##    plt.imshow(elem[0])
##    plt.show()
##    print(elem[0])
##    print(elem[1])

##ids, sequence_batch = next(iter(dataset))
##print(ids.numpy())
##print()
##print(sequence_batch.numpy())

##def process_path(file_path):
##    label = tf.strings.split(file_path, os.sep)[-2]
##    return tf.io.read_file(file_path), label
##
##labeled_ds = dataset.map(process_path)
##
##for image_raw, label_text in labeled_ds.take(1):
##    print(repr(image_raw.numpy()[:100]))
##    print()
##    print(label_text.numpy())

##iterator = iter(dataset)
##print(iterator.get_next())

#initialize model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (PIC_X, PIC_Y)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(5)
])

#compile model
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

#fit model
model.fit(dataset, epochs = 1)
