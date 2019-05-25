import numpy as np
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from utils.misc import  ImageLoader
import config

class DataLoader():
    def __init__(self, config, batch_size, seq_length, end_token=0):
        self.config = config
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token
        self.image_batch = None
        self.feature_batch = None
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        net = VGG19(weights='imagenet')
        self.trained_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def get_imagefeatures_vgg19(self, image_files, feature_files):
        #print("to extract features...")
        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, feature_files, self.batch_size) #extract image features using vgg19


    def create_batches(self, config, with_image):
        self.token_stream = []

        with open(config.temp_oracle_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

        if with_image:
            with open(config.temp_image_file) as ifile:
                self.image_files = ifile.read().splitlines()
            with open(config.temp_feature_file) as ffile:
                self.feature_files = ffile.read().splitlines()

            self.image_files = self.image_files[:self.num_batch * self.batch_size]
            self.image_batch = np.split(np.array(self.image_files), self.num_batch, 0)
            self.feature_files = self.feature_files[:self.num_batch * self.batch_size]
            self.feature_batch = np.split(np.array(self.feature_files), self.num_batch, 0)

    def create_batches_v2(self, config, with_image):
        data = np.load(config.temp_data_file).item()
        word_idxs = data['word_idxs']
        sent_lens = data['sentence_len']

        self.num_batch = int(len(word_idxs) / self.batch_size)
        word_idxs = word_idxs[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(word_idxs, self.num_batch, 0)
        self.pointer = 0

        if with_image:
            with open(config.temp_image_file) as ifile:
                self.image_files = ifile.read().splitlines()
            with open(config.temp_feature_file) as ffile:
                self.feature_files = ffile.read().splitlines()

            self.image_files = self.image_files[:self.num_batch * self.batch_size]
            self.image_batch = np.split(np.array(self.image_files), self.num_batch, 0)
            self.feature_files = self.feature_files[:self.num_batch * self.batch_size]
            self.feature_batch = np.split(np.array(self.feature_files), self.num_batch, 0)

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        imgs = None
        features = None
        if self.image_batch:
            imgs = self.image_batch[self.pointer]
            feature_files = self.feature_batch[self.pointer]
            features = self.get_imagefeatures_vgg19(imgs, feature_files)
        else:
            print("no image files")
        self.pointer = (self.pointer + 1) % self.num_batch

        return ret, features

    def reset_pointer(self):
        self.pointer = 0


class DisDataloader():
    def __init__(self, config, batch_size, seq_length):
        self.config = config
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
