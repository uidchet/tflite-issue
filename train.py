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
    def __init__(self):
        #Multiworker stratergy
        self.strategy = tf.distribute.MirroredStrategy()
        self.multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        
        self.BATCH_SIZE_PER_REPLICA= 8
        # Compute global batch size using number of replicas.
        self.global_batch_size = (self.BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync)

        
        # model and optimizer initialization
        with self.strategy.scope():
            self.depnet = depnet()
            #print(self.depnet.variables)
            # model optimizer
            self.depnet_op = tf.compat.v1.train.AdamOptimizer(0.001, beta1=0.5)
            


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
                start = time.time()
                ### train phase ###
                iterator = iter(dist_dataset)
                for step in range(step_per_epoch):
                    dep_map_loss =self.train(next(iterator))
                    

    def save(self):
        saved_model_path= '/path/to/model/'
        self.depnet.predict(np.zeros([1,256,256,6],dtype='float32'))
        #self.depnet.save(saved_model_path,save_format='tf')
        tf.saved_model.save(self.depnet,saved_model_path) 
        print("saved in pb!!!")
