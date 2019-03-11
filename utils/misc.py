
import numpy as np
import cv2
from keras.preprocessing import image
from keras import backend as K
from keras.models import Model
from scipy.linalg import norm
from keras.applications.vgg19 import VGG19
#import tensorflow as tf

#added highway for discriminator
class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)
        
    def load_image(self, file_name):
        """
        Load and preprocess an image
        """
        img = image.load_img(file_name, target_size=(224, 224))
        im = image.img_to_array(img)
        im = np.expand_dims(im, axis=0)

        im = self.preprocess_input(im)
        return im

    def preprocess_input(self, x, dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}

        if dim_ordering == 'th':
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, ::-1, :, :]
        else:
            x[:, :, :, 0] -= 103.939
            x[:, :, :, 1] -= 116.779
            x[:, :, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
        return x

    def load_image_old(self, image_file):
        """ Load and preprocess an image. """
        #print (image_file)
        image = cv2.imread(image_file)
        if (image is None):
            print ('IMAGE IS NONE :( %s' % image_file)
            return None
        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        image = image[offset[0]:offset[0]+self.crop_shape[0],
                      offset[1]:offset[1]+self.crop_shape[1]]
        image = image - self.mean
        return image

    def load_images(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file))
        images = np.array(images, np.float32)
        return images

    def extract_features(self, images, batch_size):
        #model = vgg19
        features = []
        for i in range(batch_size):
            fc2 = self.model.predict(images[i])
            #reshaped = np.reshape(fc2, (8, 512)) deleted reshape 
            features.append(reshaped)
        
        return features #shape: (batch_size, 4096)

    def mytest(self, image = 'D:/download/COCO/train/images/COCO_train2014_000000318556.jpg'):
        #model = self.vgg19

        features = []
        for i in range(1):
            #images[i] = np.expand_dims(images[i], axis=0)
            #print ('shape ' + str(images[i].shape))
            fc2 = self.model.predict(self.load_image(image))
            #reshaped = tf.reshape(fc2, [8, 512])
            features.append(reshaped)
        
        return features

def highway(input, size, num_layers=1, bias=-2.0,
            f=tf.nn.relu, scope='Highway'):
    """
        Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1-t) * y
        where g is nonlinearity, t is transform gate, and (1-t) is carry gate.
    """
    with tf.variable_scope(scope):
        size = int(size)
        for idx in range(num_layers):
            print input
            g = f(slim.fully_connected(
                input, size,
                scope='highway_lin_%d' % idx,
                activation_fn=None))

            t = tf.sigmoid(slim.fully_connected(
                input, size,
                scope='highway_gate_%d' % idx,
                activation_fn=None) + bias)

            output = t * g + (1. - t) * input
            input = output
    return output