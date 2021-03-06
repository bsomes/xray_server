from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


import numpy as np
import tensorflow as tf

model_path = './data/output_graph.pb'
labels_path = './data/output_labels.txt'
image_path = './data/'

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

    def predict(self, image):
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)

        with tf.gfile.FastGFile(image_path + image, 'rb') as im:
            image_data = im.read()

        softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        feed_dict = {'DecodeJpeg/contents:0': image_data}
        predictions = self.sess.run(softmax_tensor, feed_dict=feed_dict)
        predictions = np.squeeze(predictions)
        return JsonResponse({'prediction': [{self.labels[id]: str(val)} for (id, val) in enumerate(predictions)]})


predictor = Predictor()

def predict(request):
    response = predictor.predict(request.GET.get('image', 'B0046_0012.jpg'))
    response['Access-Control-Allow-Origin'] = '*'
    return response





