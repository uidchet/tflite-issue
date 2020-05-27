import tensorflow as tf
import tensorflow.keras.layers as layers


class depnet(tf.keras.models.Model):
    def __init__(self):
        super(depnet, self).__init__()
        filters=64
        self.img_size = 256
        kernel=(3,3)
        stride=(1,1)
        mp_stride=(2,2)

        self.conv_inp=tf.keras.layers.SeparableConv2D(filters,(3,3),strides=(1,1),padding='SAME',name='conv_input1')
        self.conv1=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_1')
        self.conv2=tf.keras.layers.SeparableConv2D(int(filters*2*1.53),kernel,strides=stride,padding='SAME',name='conv_2')
        self.conv3=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_3')
        self.conv4=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_4')
        self.conv5=tf.keras.layers.SeparableConv2D(int(filters*2*1.53),kernel,strides=stride,padding='SAME',name='conv_5')
        self.conv6=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_6')
        self.conv7=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_7')
        self.conv8=tf.keras.layers.SeparableConv2D(int(filters*2*1.53),kernel,strides=stride,padding='SAME',name='conv_8')
        self.conv9=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_9')
        self.conv10=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=stride,padding='SAME',name='conv_10')
        self.conv11=tf.keras.layers.SeparableConv2D(filters,kernel,strides=stride,padding='SAME',name='conv_11')
        self.conv12=tf.keras.layers.SeparableConv2D(1,kernel,strides=stride,padding='SAME',name='conv_12')

        self.mp_conv1=tf.keras.layers.SeparableConv2D(filters,kernel,strides=mp_stride,padding='SAME',name='conv_mp1')
        self.mp_conv11=tf.keras.layers.SeparableConv2D(filters,kernel,strides=mp_stride,padding='SAME',name='conv_mp11')
        self.mp_conv111=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=mp_stride,padding='SAME',name='conv_mp111')
        self.mp_conv2=tf.keras.layers.SeparableConv2D(filters,kernel,strides=mp_stride,padding='SAME',name='conv_mp2')
        self.mp_conv22=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=mp_stride,padding='SAME',name='conv_mp22')
        self.mp_conv3=tf.keras.layers.SeparableConv2D(filters*2,kernel,strides=mp_stride,padding='SAME',name='conv_mp3')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2= tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.bn10= tf.keras.layers.BatchNormalization()
        self.bn11 = tf.keras.layers.BatchNormalization()

    def call(self, face):

        i1=self.conv_inp(face)

        c1=self.conv1(i1)
        b1=self.bn1(c1)
        a1=tf.nn.elu(b1,name='actv_1')

        c2=self.conv2(a1)
        b2=self.bn2(c2)
        a2=tf.nn.elu(b2,name='actv_2')

        c3=self.conv3(a2)
        b3=self.bn3(c3)
        a3=tf.nn.elu(b3,name='actv_3')

        mc1= self.mp_conv1(a3)
        mc11= self.mp_conv11(mc1)
        mc111= self.mp_conv111(mc11)

        c4=self.conv4(mc1)
        b4=self.bn4(c4)
        a4=tf.nn.elu(b4,name='actv_4')

        c5=self.conv5(a4)
        b5=self.bn5(c5)
        a5=tf.nn.elu(b5,name='actv_5')

        c6=self.conv6(a5)
        b6=self.bn6(c6)
        a6=tf.nn.elu(b6,name='actv_6')

        mc2= self.mp_conv2(a6)
        mc22= self.mp_conv22(mc2)

        c7=self.conv7(mc2)
        b7=self.bn7(c7)
        a7=tf.nn.elu(b7,name='actv_7')

        c8=self.conv8(a7)
        b8=self.bn8(c8)
        a8=tf.nn.elu(b8,name='actv_8')

        c9=self.conv9(a8)
        b9=self.bn9(c9)
        a9=tf.nn.elu(b9,name='actv_9')

        mc3= self.mp_conv3(a9)

        x= tf.concat([mc111,mc22,mc3],axis=-1)

        c10=self.conv10(x)
        b10=self.bn10(c10)
        a10=tf.nn.elu(b10,name='actv_10')

        c11=self.conv11(a10)
        b11=self.bn11(c11)
        a11=tf.nn.elu(b11,name='actv_11')

        c12=self.conv12(a11)
        a12=tf.nn.elu(c12,name='actv_12')

        return a12



