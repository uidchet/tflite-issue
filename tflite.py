import tensorflow as tf

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
