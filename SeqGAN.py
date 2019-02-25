import tensorflow as tf
import numpy as np
from Discriminator import Discriminator
from generator import Generator

import os
import shutil

#from Shaofan Lai's tensorflow implementation of SeqGAN (Yu et. al, 2017)
#https://github.com/Shaofanl/SeqGAN-Tensorflow
#SeqGAN: https://arxiv.org/abs/1609.05473

#Audrey's modifications:
#removed parameters for seqgan constructor, generator, disciminator (parameters are already in config)
class SeqGAN(object):
    def __init__():

        '''

        # generator related
        g_emb_dim: the size of the embedding space
        g_hidden_dim: the number of the hidden cells in LSTM 

        # discriminator related
        d_emb_dim: the size of the embedding space
        d_filter_sizes: the size of the filters (CNN)
        d_num_filters: the number of the filters (CNN)

        # others
        log_generation: whether to log the generation as
            an image in the tensorboard'''
        self.generator = Generator()
        self.discriminator = Discriminator() 
        self.log_generation = log_generation
        self.config = Config()

    def train(self, data):
    '''
        sampler: a function to sample given batch_size

        evaluator: a function to evaluate given the
            generation and the index of epoch
        evaluate: a bool function whether to evaluate
            while training
    '''

        # if os.path.exists(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        # os.mkdir(tensorboard_dir)

        config = self.config
        pretrain_g_epochs = config.pretrain_g_epochs
        pretrain_g_epochs = 1 #make this 1 for debugging purposes
        pretrain_d_epochs = config.pretrain_d_epochs
        pretrain_d_epochs = 1 #make this 1 for debugging purposes
        tensorboard_dir = 'tensorboard/' #TODO check what this is
        #TODO check what sampler is
        gen, dis = self.generator, self.discriminator
        batch_size = config.batch_size

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            print ('pretraining Generator ...')
            for epoch in range(pretrain_g_epochs):
                print ('pretrain g epoch', epoch) #sampler gets coco data 
                summary = gen.pretrain(sess, sampler(config.batch_size)) #pretrain generator 
                writer.add_summary(summary, epoch)

                if evaluate and evaluator is not None:
                    evaluator(gen.generate(sess), epoch)

            print ('pretraining Discriminator ...')
            for epoch in range(pretrain_d_epochs):
                fake_samples = gen.generate(sess)
                real_samples = sampler(config.batch_size) #TODO change sampler 
                samples = np.concatenate([fake_samples, real_samples])
                labels = np.concatenate([np.zeros((batch_size,)),
                                         np.ones((batch_size,))])
                for _ in range(3):
                    indices = np.random.choice(
                        len(samples), size=(batch_size,), replace=False)
                    dis.train(sess, samples[indices], labels[indices])

            print( 'Start Adversarial Training ...')
            for epoch in range(config.total_epochs):
                print 'epoch', epoch
                for _ in range(1):
                    fake_samples = gen.generate(sess) #generator generates fake samples
                    rewards = gen.get_reward(sess, fake_samples, 16, dis) 
                    summary = gen.train(sess, fake_samples, rewards) #generate new fake samples and reward
                    # np.set_printoptions(linewidth=np.inf,
                    #                     precision=3)
                    # print rewards.mean(0)
                writer.add_summary(summary, epoch)

                for _ in range(5):
                    fake_samples = gen.generate(sess) #generator generates fake samples after being trained
                    real_samples = sampler(batch_size) #takes samples from real data
                    samples = np.concatenate([fake_samples, real_samples])
                    labels = np.concatenate([np.zeros((batch_size,)),
                                             np.ones((batch_size,))])
                    for _ in range(3):
                        indices = np.random.choice(
                            len(samples), size=(batch_size,), replace=False)
                        summary = dis.train(sess, samples[indices],
                                            labels[indices]) #discriminator trains on the fake and real samples
                writer.add_summary(summary, epoch)

                if self.log_generation:
                    summary = sess.run(
                        gen.image_summary,
                        feed_dict={gen.given_tokens: real_samples})
                    writer.add_summary(summary, epoch)

                if evaluate and evaluator is not None:
                    evaluator(gen.generate(sess), pretrain_g_epochs+epoch)

                np.save('generation', gen.generate(sess))
