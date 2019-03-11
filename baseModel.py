import copy
import json
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


import six.moves.cPickle as pickle
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import  ImageLoader
from utils.nn import NN

from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

import mxnet as mx
import cv2
from img2poem.symbol_vgg import VGG
import img2poem.symbol_sentiment
import config

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)

        self.build()

    def get_imagefeatures(self, image_files, batch_size):
        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, batch_size)

    def build(self):
        # use pretrained vgg model to extract image features
        net = VGG19(weights='imagenet')
        self.trained_model = Model(input= net.input, output= net.get_layer('fc2').output)

        self.object_model = self.get_mod()
        self.object_model.load_params('./img2poem/model/object.params')

        self.sentiment_model = self.get_mod(sym = img2poem.symbol_sentiment.get_sym(), img_len = 227)
        self.sentiment_model.load_params('./img2poem/model/Sentiment.params')

        self.scene_model = self.get_mod()
        self.scene_model.load_params('./img2poem/model/scene.params')
        #self.sentiment_model = Model(input= net.input, output= net.get_layer('block3_conv1').output)

    def get_mod(self, output_name = 'relu7_output', sym = None, img_len = 224):
        if sym is None:
            vgg = VGG()
            sym = vgg.get_symbol(num_classes = 1000, 
                      blocks = [(2, 64),
                                (2, 128),
                                (3, 256), 
                                (3, 512),
                                (3, 512)])
            internals = sym.get_internals()
            sym = internals[output_name]
        ctx = mx.cpu()
        mod = mx.module.Module(
                context = ctx,
                symbol = sym,
                data_names = ("data", ),
                label_names = ()
        )

        mod.bind(data_shapes = [("data", (1, 3, 224, 224))], for_training = False)

        return mod

    def train(self, sess, train_data):
        raise NotImplementedError()

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
        #for k in range(1):
            batch = eval_data.next_batch()
            #caption_data = self.beam_search(sess, batch, vocabulary)
            images = self.image_loader.load_images(batch)
            caption_data, scores = sess.run([self.predictions, self.probs], feed_dict={self.images: images})
            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = str(vocabulary.get_sentence(word_idxs))
                results.append({'image_id': int(eval_data.image_ids[idx]),
                                'caption': caption})
                #print(results)
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = mpimg.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            images = self.image_loader.load_images(batch)
            caption_data,scores_data = sess.run([self.predictions,self.probs],feed_dict={self.images:images})

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = vocabulary.get_sentence(word_idxs)
                print(caption)
                captions.append(caption)
                scores.append(scores_data[l])

                # Save the result in an image file
                image_file = batch[l]
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]
                img = mpimg.imread(image_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(config.test_result_dir,
                                         image_name+'_result.jpg'))

        ##Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("All variables present...")
        for var in tf.all_variables():
            print(var)
        with tf.variable_scope('conv1_1',reuse = True):
            kernel = tf.get_variable('conv1_1_W')

        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path,encoding='latin1')
        count = 0
        for param_name in tqdm(data_dict.keys()):
            op_name = param_name[:-2]
            print(param_name)
            #print(op_name)
            with tf.variable_scope(op_name, reuse = True):
                try:
                    var = tf.get_variable(param_name)
                    session.run(var.assign(data_dict[param_name]))
                    count += 1
                except ValueError:
                    print("No such variable")
                    pass

        print("%d tensors loaded." %count)