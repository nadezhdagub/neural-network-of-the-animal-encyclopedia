import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])


def test_digit(sample):
    sample = sample[np.newaxis, ...]
    prediction = loaded_model.predict(sample)
    # print(prediction)
    ans = np.argmax(prediction)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(sample.reshape(150, 150, 3),
              cmap=matplotlib.cm.binary, interpolation='none')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    ax = fig.add_subplot(1, 2, 2)
    bar_list = ax.bar(np.arange(2), prediction[0], align='center')  # //////
    bar_list[ans].set_color('g')
    ax.set_xticks(np.arange(2))
    ax.set_xlim([-1, 10])
    ax.grid('on')

    # plt.show()

    print('{}'.format(ans))
    # print('Answer: {}'.format(ans))

from PIL import Image

import sys

img = Image.open(sys.argv[1:][0]).convert('RGB')

img = img.resize((150,150), Image.LANCZOS)
img = np.array(img)
test_digit(img)
