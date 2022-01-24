import tensorflow as tf
import os

def conventer(savedir, op_name):
    """
    op_name should be { flows | interpolated }
    """
    converter = tf.lite.TFLiteConverter.from_saved_model("%s/trained model/%s" % (savedir,op_name))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open(savedir+"/"+op_name+".tflite", "wb").write(tflite_model)


if __name__=="__main__":
    savedir = os.path.dirname(os.path.abspath(__file__))+"/results/juice/"
    conventer(savedir, "flows")
    conventer(savedir, "interpolated")