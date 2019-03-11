import datetime
import time
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):

        #added time stamp to save directories 
        ts = time.time()
        self.st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

        # random seed
        self.seed = 17

        # about the model architecture
        self.cnn = 'vgg19'               # changed from vgg16 to vgg19
        self.max_caption_length = 70
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 1 ## Changed from 2 to 1    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 1    ## Changed from 2 to 1   # 1 or 2
        self.dim_decode_layer = 1024
        self.image_feat_dim = 4096*1 #for both image object, sentiment, and scene features
        self.G_hidden_size = 512
        self.START = 0

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.5
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 200
        self.total_epochs = 200 #added for gan
        self.pretrain_g_epochs=1000 #added for gan
        self.pretrain_d_epochs=50 #added for gan
        self.d_filter_sizes=[3, 5, 5, 5] #added for discriminator
        self.d_num_filters=[50, 80, 80, 100] #added for discriminator
        self.highway_layers = 5 # added for dis
        self.batch_size = 16
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.learning_rate = 0.001 # for seqgan
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 1000
        self.save_dir = 'D:/dev/show_and_tell/models_gan_art/'
        self.summary_dir = 'D:/dev/show_and_tell/summary_gan_art' + self.st+'/'
        self.log_dir = 'D:/dev/show_and_tell/logs_gan_art/'
        self.eval_log_dir = 'D:/dev/show_and_tell/elogs/' #TODO eval

        # about the training

        # Dataset - COCO
        #base_dir = 'D:/download/COCO'
        #self.ignore_file = base_dir + '/ignore.csv'
        #self.train_caption_file = base_dir + '/train/captions_train2014.json'
        #self.eval_caption_file = base_dir + '/val/captions_val2014.json'
        #self.vocabulary_size = 5000

        # Dataset - art description
        base_dir = 'D:/download/art_desc'
        self.train_caption_file = base_dir + '/train/ann.csv'
        self.eval_caption_file = base_dir + '/val/ann.csv'
        self.vocabulary_size = 10000

        self.vocabulary_file = base_dir + '/vocabulary.csv'
        self.ignore_file = base_dir + '/ignore.csv'
        self.ignore_file_eval = base_dir + '/ignore_eval.csv'

        self.train_image_dir = base_dir + '/train/images/'
        self.temp_annotation_file = base_dir + '/train/ann.csv'
        self.temp_data_file = base_dir + '/train/data.npy'

        # about the evaluation
        self.eval_image_dir = base_dir + '/val/images/'
        self.eval_result_dir = base_dir + '/val/'
        self.eval_result_file = base_dir + '/val/results' + self.st + '.json'
        #self.eval_result_file = base_dir + '/val/results' + 'forfun' + '.json'
        self.save_eval_result_as_image = False

        # about the testing
        self.test_image_dir = base_dir + './test/images/'
        self.test_result_dir = base_dir + './test/results/'
        self.test_result_file = base_dir + './test/results.csv'

        self.trainable_variable = False