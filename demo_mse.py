import argparse
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='device=gpu,floatX=float32'

import numpy as np

import glob
import fnmatch

import cv2

from keras import backend as K
K.set_image_dim_ordering('th')
K.set_epsilon(1e-7)

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

#from model import VGG_16

import theano
theano.config.openmp = True



def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations


def loadImages(folder):
    imgs = []
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))
   
    return sorted(matches)

batch_size = 1


def loadModel():
# load json and create model
    json_file = open('./erika.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./erika_unstable.h5")
    return model


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--split_num', type=int, default=1)
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()
    assert args.split_num > args.split >= 0
    assert args.split_num > 0

    model = loadModel()
    imnames = loadImages(args.input_dir)
    # split imnames into split_num pieces to run in multi-thread
    imnames = imnames[args.split::args.split_num]

    for imname in imnames:
        print(imname)
        src = cv2.imread(imname,cv2.IMREAD_GRAYSCALE)

        rows = int(src.shape[0]/16 + 1)*16
        cols = int(src.shape[1]/16 + 1)*16

        patch = np.empty((1,1,rows,cols),dtype="float32")
        patch[0,0,:,:] = np.ones((rows,cols),dtype="float32")*255.0
        patch[0,0,0:src.shape[0],0:src.shape[1]] = src
        out = model.predict(patch, batch_size=batch_size)
        if isinstance(out, list):
            out = out[0]

        result = np.zeros((rows,cols),dtype=np.float32)

        result = out[0,0,:,:] 
    
        print(np.amax(result), np.amin(result))

        result2 = cv2.normalize(result,0,255)

        head, tail = os.path.split(imname)
        result[result>255] = 255
        result[result<0] = 0
        cv2.imwrite(args.output_dir+"/"+tail+".png", result[0:src.shape[0],0:src.shape[1]])

if __name__ == "__main__":
    test()
