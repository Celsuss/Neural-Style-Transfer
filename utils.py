import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import os

mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

def load_img(path):
    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def save_img(image, path, name):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError as exc:
            print('Can\'t create path: {}'.format(os.path.join(os.getcwd(), path)))
            return
            
    full_path = os.path.join(path, name+'.png')
    print('Saving image {}'.format(full_path))
    mpl.image.imsave(full_path, image[0])

def imshow(image, title=None, draw=True):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    
    if draw:
        plt.show()

def draw():
    plt.show()