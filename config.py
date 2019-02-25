
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'vgg19'               # changed from vgg16 to vgg19
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 1 ## Changed from 2 to 1    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 1    ## Changed from 2 to 1   # 1 or 2
        self.dim_decode_layer = 1024
        self.image_feat_dim = 4096
        self.G_hidden_size = 512
        self.START = 0

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 200
        self.total_epochs = 200 #added for gan
        self.pretrain_g_epochs=1000 #added for gan
        self.pretrain_d_epochs=50 #added for gan
        self.d_filter_sizes=[3, 5, 5, 5], #added for discriminator
        self.d_num_filters=[50, 80, 80, 100] #added for discriminator
        self.batch_size = 32
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
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
        self.save_dir = 'D:/dev/show_and_tell/models_gan/'
        self.summary_dir = 'D:/dev/show_and_tell/summary_gan/'

        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'
        self.vocabulary_size = 5000

        # about the training
        base_dir = 'D:/download/COCO'
        
        self.train_image_dir = base_dir + '/train/images/'
        self.train_caption_file = base_dir + '/train/captions_train2014.json'
        self.temp_annotation_file = base_dir + '/train/anns.csv'
        self.temp_data_file = base_dir + '/train/data.npy'

        # about the evaluation
        self.eval_image_dir = base_dir + '/val/images/val2014/'
        self.eval_caption_file = base_dir + '/val/captions_val2014.json'
        self.eval_result_dir = base_dir + '/val/results/'
        self.eval_result_file = base_dir + '/val/results.json'
        self.save_eval_result_as_image = False

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'

        self.trainable_variable = False