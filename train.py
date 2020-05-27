from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
#from loss import euc_loss
from networks import depnet
from utils import Error,plotResults,Generator,PlotLosses
import time
import random
import math
import tensorflow_addons as tfa
import numpy as np

class Model:
    def __init__(self, config):
        self.config = config
        #Multiworker stratergy
        self.strategy = tf.distribute.MirroredStrategy()
        self.multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # model losses
        # self.depth_map_loss = Error()
        # model saving setting
        self.last_epoch = 0
        self.checkpoint_manager = []
        # loss function
        #self.fl=euc_loss()
        # batch per replica
        self.BATCH_SIZE_PER_REPLICA=self.config.BATCH_SIZE_PER_REPLICA
        # Compute global batch size using number of replicas.
        self.global_batch_size = (self.BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync)

        #checkpoint
        self.checkpoint_dir = os.path.join(self.config.LOG_DIR,self.config.SESSION_ID)

        # model and optimizer initialization
        with self.strategy.scope():
            self.depnet = depnet()
            #print(self.depnet.variables)
            # model optimizer
            self.depnet_op = tf.compat.v1.train.AdamOptimizer(config.LEARNING_RATE, beta1=0.5)
            # self.checkpoint = tf.train.Checkpoint(depnet=self.depnet,depnet_optimizer=self.depnet_op)
            # self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=30)
            # last_checkpoint = self.checkpoint_manager.latest_checkpoint
            # self.checkpoint.restore(last_checkpoint)
            # if last_checkpoint:
            #     self.last_epoch = int(last_checkpoint.split('-')[-1])
            #     print("Restored from {}".format(last_checkpoint))
            # else:
            #     print("Initializing from scratch.")


    @tf.function
    def train(self, dataset_train):
        def one_step(data_batch):
            image,dmap = data_batch
            with tf.GradientTape() as tape:
                dmap_pred= self.depnet.call(image)
                # supervised feature loss
                ypred=tf.reshape(dmap_pred,[dmap_pred.get_shape()[0],-1])
                ytrue=tf.reshape(dmap,[dmap.get_shape()[0],-1])
                l2_norm = tf.keras.losses.MSE(ytrue, ypred)
                depth_map_loss = l2_norm* (1.0 / self.global_batch_size)

            # back-propagate
            gradients = tape.gradient(depth_map_loss, self.depnet.variables)
            self.depnet_op.apply_gradients(zip(gradients, self.depnet.variables))


            return depth_map_loss

        per_example_losses =self.strategy.run(one_step,args=(dataset_train,))
        depth_map_loss=self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

        return depth_map_loss

    def start_tarining(self):

        # Get images and labels.
        image_gen=Generator("train")

        #distributed dataset
        dataset = tf.data.Dataset.from_generator(image_gen.generator,output_types=(tf.float32,tf.float32),output_shapes=(tf.TensorShape([256, 256,6]),tf.TensorShape([32,32,1])))
        dataset = dataset.batch(self.global_batch_size)
        dist_dataset= self.strategy.experimental_distribute_dataset(dataset)

        step_per_epoch=int(math.floor(len(image_gen.data_list_with_labels)/self.global_batch_size))

        # initialize loss plots in mlflow
        mlflow_plot=PlotLosses()

        # begin training
        with self.strategy.scope():
            for epoch in range(0,1):
                step_per_epoch=int(math.floor(len(image_gen.data_list_with_labels)/self.global_batch_size))
                start = time.time()
                ### train phase ###
                iterator = iter(dist_dataset)
                for step in range(step_per_epoch):
                    dep_map_loss =self.train(next(iterator))
                    # avg_loss=self.depth_map_loss(dep_map_loss)
                    # display loss
                    # print('Epoch {:d}-{:d}/{:d}: Map:{:.6f}'.format(epoch + 1, step + 1, step_per_epoch,avg_loss), end='\r')
                # #plot losses
                # mlflow_plot.plot(epoch,avg_loss,self.config.MLFLOW_ARTIFACTS_DIR)
                # # time of one epoch
                # print('\n    Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))
                # # checkpoint save
                # #if (epoch + 1) % 1 == 0:
                #  #   self.checkpoint_manager.save(checkpoint_number=epoch + 1)
                # print('\n', end='\r')
                # self.depth_map_loss.reset()
                # random.shuffle(image_gen.data_list_with_labels)
                # if (epoch+1)%16==0:
                #     #increase batch size by 10%
                #     self.global_batch_size=int( self.global_batch_size+ self.global_batch_size*10/100)
                #     if self.global_batch_size%2 !=0:
                #         self.global_batch_size=self.global_batch_size-1

    def save(self):
        saved_model_path= os.path.join(self.config.LOG_DIR,self.config.SESSION_ID,str(epoch))
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        self.depnet.predict(np.zeros([1,256,256,6],dtype='float32'))
        #self.depnet.save(saved_model_path,save_format='tf')
        tf.saved_model.save(self.depnet,saved_model_path) 
        print("saved in pb!!!")
