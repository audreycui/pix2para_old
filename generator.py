import tensorflow as tf
import numpy as np
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from baseModel import BaseModel

#Audrey's modifications: 
    #added build_vgg19 method using keras
    #added lstm encoder
    #added batch normalization method
    #removed reduce_mean of convnet features
    #changed argmax to monte carlo for sampling next word when training rnn
    #moved build functions (cnn, rnn, optimizer) to constructor
    #added pretrain function
    #added generate function 
class Generator(BaseModel):
    def __init__(self): #replace build
        
        #self.is_train = true for now since I'm working on training my models
        #build cnn for image features (vgg19)
        print("Building the CNN...")
        config = self.config
        self.net = VGG19(weights='imagenet')
        self.model = Model(input= self.net.input, output= self.net.get_layer('fc2').output)
        print("CNN built.")

        #Audrey's modifications
        #changed rnn decoder to lstm image encoder + lstm decoder
        #new decoder takes in the encoder's final state as its initial state (rather than initial token)
        print("Building the RNN...")
        config = self.config
        #added in config: image_feature_dim = 4096, G_hidden_size = 512

        # Setup the placeholders
        if self.is_train:
            #contexts = self.conv_feats
            self.conv_feats = tf.placeholder(dtype = tf.float32, shape = [config.batch_size, config.image_feat_dim]) #added 
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])

        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            #tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("image_feat", reuse = tf.AUTO_REUSE):
                # name: "generator/image_feat"
                image_feat_W = tf.get_variable("image_feat_W", [config.image_feat_dim, config.G_hidden_size], tf.float32, random_uniform_init)
                image_feat_b = tf.get_variable("image_feat_b", [config.G_hidden_size], tf.float32, random_uniform_init)
            
            with tf.variable_scope("output"):
                # name: "generator/output"
                output_W = tf.get_variable("output_W", [config.G_hidden_size, config.vocab_size], tf.float32, random_uniform_init)
                output_b = tf.get_variable("output_b", [config.vocab_size], tf.float32, random_uniform_init)
            
        # Setup the word embedding, we can use pre trained word embeddings later //TODO
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        #added LSTM to encode image features
        #setup the lstm encoder 
        with tf.variable_scope("lstm_encoder"):
            encoder = tf.nn.rnn_cell.LSTMCell(config.G_hidden_size, state_is_tuple=True)


        # Setup the LSTM decoder
        with tf.variable_scope("lstm_decoder"):
            decoder = tf.nn.rnn_cell.LSTMCell(
                config.G_hidden_size,
                initializer = self.nn.fc_kernel_initializer)
            if self.is_train:
                decoder = tf.nn.rnn_cell.DropoutWrapper(
                    decoder,
                    input_keep_prob = 1.0-config.lstm_drop_rate,
                    output_keep_prob = 1.0-config.lstm_drop_rate,
                    state_keep_prob = 1.0-config.lstm_drop_rate)



        ## 8 * 512 is reduced to 512(embedding size of word) by mean, We can use matrix multiplication also to covert // TODO
        #initial_memory = tf.zeros([config.batch_size, lstm.state_size[0]])
        #initial_output = tf.zeros([config.batch_size, lstm.state_size[1]])

        # Prepare to run
        predictionsArr = [] #indices of predictions
        output_probs = [] #probabilities of output
        cross_entropies = []
        predictions_correct = []
        num_steps = config.max_caption_length
        #image_emb = tf.reduce_mean(self.conv_feats, axis =1) #removed reduce mean
        
        #============================= encoder ===================================================================
        #input conv_feats dimensions: (batch_size, 4096)
        state = encoder.zero_state(config.batch_size, tf.float32)
        with tf.variable_scope("image_feat") as scope:
            image_feat = self.batch_norm(self.conv_feats[:,:], mode='test', name='')
        image_feat_emb = tf.matmul(self.conv_feats, image_feat_W) + image_feat_b  # B,H
        lstm_input = image_feat_emb
        with tf.variable_scope("lstm_encoder") as scope:
            _, state = encoder(lstm_input, state)
            encoder_state = state 

        ## Initial memory and output are given zeros
        #last_memory = initial_memory
        #last_output = initial_output
        last_word = image_feat_emb

        #============================= decoder ===================================================================
        #last_state = last_memory, last_output
        last_state = encoder_state #modified: initial state of decoder is the final state of encoder

        start_token = tf.constant(config.START, tf.int32, [config.batch_size])
        # Generate the words one by one
        for idx in range(num_steps):
            #first step: start token. otherwise, embed last word
            if idx == 0:
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                        start_token)
            else:
                with tf.variable_scope("word_embedding"):
                    word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                        last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                if not idx == 0:
                    tf.get_variable_scope().reuse_variables()

                output, state = decoder(word_embed, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                ## Logits is of size vocab
                logits = tf.matmul(output, output_W) + output_b
                log_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-20, 1.0))  # B,Vocab_size # add 1e-8 to prevent log(0)
                output_probs.append(log_probs)
                self.probs = log_probs
                # sample once from the multinomial distribution
                # Montecarlo sampling
                prediction = tf.reshape(tf.multinomial(log_probs, 1), [config.batch_size])   # 1 means sample once
                #prediction is the index of predicted word
                predictionsArr.append(prediction)

                #deleted calculation of loss function
            last_state = state
            if self.is_train:
                ##  During training the input to LSTM is fed by user
                last_word = sentences[:, idx]
            else:
                # During testing the input to current time stamp of LSTM is the previous time stamp output.
                last_word = prediction

            tf.get_variable_scope().reuse_variables()

        #TODO: figure out the purpose of the loss calculation below: 
        if self.is_train:
            # Compute the final loss, if necessary
            '''
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis=1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)
            '''
            pretrain_loss = tf.reducemean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = given_tokens, logits = output_probs))
            pretrain_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
            pretrain_op = slim.learning.create_train_op(
                pretrain_loss, pretrain_optimizer, clip_gradient_norm=5.0)

            self.pretrain_loss = pretrain_loss
            self.pretrain_op = pretrain_op
            self.pretrain_summary = tf.summary.scalar(
                "g_pretrain_loss", pretrain_loss)

        #reinforcement learning
        rewards = tf.placeholder(tf.float32,
                             shape=[batch_size, num_steps],
                             name="rewards")
        g_seq = self.output_ids[num_steps]  # follow the generated one
        g_prob = self.output_probs[num_steps]
        g_loss = -tf.reduce_mean(
            tf.reduce_sum(tf.one_hot(g_seq, config.vocabulary_size) * safe_log(g_prob), -1) *
            rewards
        )
        g_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        g_op = slim.learning.create_train_op(
            g_loss, g_optimizer, clip_gradient_norm=5.0)
        g_summary = tf.summary.merge([
            tf.summary.scalar("g_loss", g_loss),
            tf.summary.scalar("g_reward", tf.reduce_mean(rewards))
        ])
 
        self.rewards = rewards
        self.g_op = g_op
        self.g_summary = g_summary
        self.image_summary = tf.summary.merge([
            tf.summary.image(
                "real_samples",
                tf.expand_dims(tf.one_hot(given_tokens, config.vocabulary_size), -1)
            ),
            tf.summary.image(
                "fake_samples",
                tf.expand_dims(tf.one_hot(output_ids[0], config.vocabulary_size), -1)
            ),
        ])
            #TODO Add reward!!
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
        self.predictions = tf.stack(predictionsArr, axis=1)

        print("RNN built.")

    def generate(self, sess):
        return sess.run(self.predictions)

    #removed build_cnn, build_rnn, moved it into the constructor
    
    #TODO: fix this mess!!! given_tokens and images and saving 
    def pretrain(self, sess, given_tokens):
        images = self.image_loader.load_images(image_files)

        conv_features = self.extract_features(images, self.config.batch_size) #extract image features using vgg19

        feed_dict = {self.sentences: sentences, #removed images bc already got image features
                     self.masks: masks, 
                     self.conv_feats: conv_features}
        feed_dict = {self.given_tokens: given_tokens}
        _, summary = sess.run([self.pretrain_op, self.pretrain_summary],
                              feed_dict=feed_dict)
        return summary

    def pretrain_old(self, sess, train_data): #changed train method to pretraining method for generator
    """ Pretrain the generator using the COCO train2014 data. """

        #TODO move this out of basemodel and into generator, create more general training/pretraining method here
        self.build_optimizer() # added build optimizer in the training method
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)

                conv_features = self.extract_features(images, self.config.batch_size) #extract image features using vgg19


                feed_dict = {self.sentences: sentences, #removed images bc already got image features
                             self.masks: masks, 
                             self.conv_feats: conv_features}
                # _, summary, global_step = sess.run([self.opt_op,
                #                                     self.summary,
                #                                     self.global_step],
                #                                     feed_dict=feed_dict)
                _, global_step = sess.run([self.opt_op,
                                                    self.global_step],
                                                   feed_dict=feed_dict)
                if (global_step + 1) % config.save_period == 0:
                    self.save()
                #train_writer.add_summary(summary, global_step)
            train_data.reset()

        self.save()
        train_writer.close()
        print("Training complete.")

    def extract_features(self, images, batch_size):
        #model = vgg19
        features = []
        for i in range(batch_size):
            fc2 = self.model.predict(images[i])
            reshaped = np.reshape(fc2, (4096))  
            features.append(reshaped)
        
        return features #shape: (batch_size, 4096)
        #added batch normalization helper method

    def batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)


        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


