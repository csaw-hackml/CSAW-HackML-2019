from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import h5py
import numpy as np
import sys


def usage():
    print('Usage: %s data.h5 model.h5' % sys.argv[0])

def evaluate():
    if len(sys.argv) != 3:
        usage()
        exit(0)

    data_path = sys.argv[1]
    model_path = sys.argv[2]

    # load data
    print('loading data...')
    data = h5py.File(data_path)
    data_x = data['data']
    data_y = np.array(data['label'])

    datagen_test = ImageDataGenerator(
        rescale=1/255.0
    )

    # load model
    print('loading model...')
    m = load_model(model_path)

    print('\n\n')

    # clean data accuracy
    p = np.argmax(m.predict_generator(datagen_test.flow(data_x, shuffle=False, batch_size=1), len(data_x)), axis=1)
    clean_data_acc = (np.sum(p == data_y) / float(len(data_y)))
    print('Clean data accuracy: %f' % clean_data_acc)


if __name__=='__main__':
    evaluate()
    