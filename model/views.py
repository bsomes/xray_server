from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


import numpy as np
import tensorflow as tf

model_path = './data/output_graph.pb'
labels_path = './data/output_labels.txt'
image_path = './data/B0046_0012.jpg'

class Predictor(object):
    def __init__(self):
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')


        f = open(labels_path, 'r')
        lines = f.readlines()
        self.labels = [str(w).replace("\n", "") for w in lines]
        self.sess = tf.Session()

    def predict(self):
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)

        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        mul = self.sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
        predictions = self.sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        return JsonResponse({'prediction': [{self.labels[id]: str(val)} for (id, val) in enumerate(predictions)]})


predictor = Predictor()

def predict(request):
    return predictor.predict()


if __name__ == '__main__':
    print (predict(None))



