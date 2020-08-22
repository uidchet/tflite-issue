import tensorflow as tfimport tensorflow as tf
import numpy as np
class Dep_wise_Conv(tf.keras.Model):
    def __init__(self,filters,kernel=(3,3),stride=(1,1),_dep_multiplier=1,_name="default"):
        super(Dep_wise_Conv, self).__init__()
        initializer = tf.keras.initializers.he_normal()
        self.stride=stride
        self.name2=_name
        self.dep_conv1=tf.keras.layers.DepthwiseConv2D(kernel,strides=(1,1),padding='SAME',depth_multiplier=_dep_multiplier,name='dep_conv_'+self.name2,depthwise_initializer=initializer)
        self.bn_dep1 = tf.keras.layers.BatchNormalization()
        self.point_conv1=tf.keras.layers.Conv2D(filters,(1,1),strides=self.stride,padding='SAME',name='point_conv_'+self.name2,kernel_initializer=initializer)
        self.bn_point1 = tf.keras.layers.BatchNormalization()
    def call(self,inp):
        _op=self.dep_conv1(inp)
        _op=self.bn_dep1(_op)
        _op=tf.nn.leaky_relu(_op,name='dep_leaky_relu_'+self.name2)
        _op=self.point_conv1(_op)
        _op=self.bn_point1(_op)
        _op=tf.nn.leaky_relu(_op,name='point_leaky_relu_'+self.name2)
        if inp.shape[-1] == _op.shape[-1] and inp.shape[1] == _op.shape[1]:
            _op=inp+_op
        return _op


class depnet(tf.keras.Model):
    def __init__(self):
        super(depnet, self).__init__()
        filters=128
        kernel=(3,3)
        stride=(1,1)
        mp_stride=(2,2)
        self.conv_inp=tf.keras.layers.Conv2D(filters,(3,3),strides=(1,1),padding='SAME',name='conv_input1')
        self.conv1=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_1')
        self.conv2=Dep_wise_Conv(int(filters*2*2),kernel,stride=stride,_name='conv_2')
        self.conv3=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_3')
        self.conv4=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_4')
        self.conv5=Dep_wise_Conv(int(filters*2*2),kernel,stride=stride,_name='conv_5')
        self.conv6=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_6')
        self.conv7=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_7')
        self.conv8=Dep_wise_Conv(int(filters*2*2),kernel,stride=stride,_name='conv_8')
        self.conv9=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_9')
        self.conv10=Dep_wise_Conv(filters*2,kernel,stride=stride,_name='conv_10')
        self.conv11=Dep_wise_Conv(filters,kernel,stride=stride,_name='conv_11')
        self.conv12=Dep_wise_Conv(1,kernel,stride=stride,_name='conv_12')

        self.mp_conv1=Dep_wise_Conv(filters,kernel,stride=mp_stride,_name='conv_mp1')
        self.mp_conv11=Dep_wise_Conv(filters,kernel,stride=mp_stride,_name='conv_mp11')
        self.mp_conv111=Dep_wise_Conv(filters*2,kernel,stride=mp_stride,_name='conv_mp111')
        self.mp_conv2=Dep_wise_Conv(filters,kernel,stride=mp_stride,_name='conv_mp2')
        self.mp_conv22=Dep_wise_Conv(filters*2,kernel,stride=mp_stride,_name='conv_mp22')
        self.mp_conv3=Dep_wise_Conv(filters*2,kernel,stride=mp_stride,_name='conv_mp3')


        self.bn_inp = tf.keras.layers.BatchNormalization()
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
        
    def call(self, inputs):
        i1=self.conv_inp(inputs)
        ai=tf.nn.leaky_relu(i1)
        bi1=self.bn_inp(ai)

        c1=self.conv1(bi1)
        b1=self.bn1(c1)
        c2=self.conv2(b1)
        b2=self.bn2(c2)
        c3=self.conv3(b2)
        b3=self.bn3(c3)
        

        mc1= self.mp_conv1(b3)
        mc11= self.mp_conv11(mc1)
        mc111= self.mp_conv111(mc11)

        c4=self.conv4(mc1)
        b4=self.bn4(c4)

        c5=self.conv5(b4)
        b5=self.bn5(c5)

        c6=self.conv6(b5)
        b6=self.bn6(c6)

        mc2= self.mp_conv2(b6)
        mc22= self.mp_conv22(mc2)

        c7=self.conv7(mc2)
        b7=self.bn7(c7)

        c8=self.conv8(b7)
        b8=self.bn8(c8)

        c9=self.conv9(b8)
        b9=self.bn9(c9)

        mc3= self.mp_conv3(b9)

        x= tf.concat([mc111,mc22,mc3],axis=-1)

        c10=self.conv10(x)
        b10=self.bn10(c10)

        c11=self.conv11(b10)
        b11=self.bn11(c11)

        c12=self.conv12(b11)
        return c12
       
       
model_=depnet()
x=np.zeros((1,256,256,6))
sdf=model_.predict([x,x1,x2,x3])
model_.summary()
tf.saved_model.save(model_,"/path/") #also tried model_.save


def convert_model(path,tflite_path):
    tf.keras.backend.clear_session()

    #concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #concrete_func.inputs[0].set_shape(Config.INPUT_SHAPE)
    #converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.allow_custom_ops = True

    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)
    tf.keras.backend.clear_session()

convert_model('/path/','/path/x.tflite')#first arg is path to saved model
