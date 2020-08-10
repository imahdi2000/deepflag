import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt      #Comment this and lines 57-61 out if you don't have matplot innstalled

TRAIN_DATA_REPETITION = 10
PIC_X = 28
PIC_Y = 28

#import images
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#output
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#preprocess
train_images = train_images / 255.0
test_images = test_images / 255.0


#initialize model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(PIC_X, PIC_Y)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
])

#compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train model
model.fit(train_images, train_labels, epochs=TRAIN_DATA_REPETITION)

#Test Accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)



#Making Predictions
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

TEST_NUM = 100
print (predictions[TEST_NUM])
print("Prediction:", class_names[np.argmax(predictions[TEST_NUM])])
print("Actual Answer:", class_names[test_labels[TEST_NUM]])

#======Display Image=====
# plt.figure()
# plt.imshow(test_images[TEST_NUM])
# plt.colorbar()
# plt.grid(False)
# plt.show()
#========================


# img = test_images[1]
# print(img.shape)
# predictions_single = probability_model.predict(img)

# print(predictions_single)

# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)

# np.argmax(predictions_single[0])
