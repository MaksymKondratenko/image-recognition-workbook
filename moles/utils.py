import json
import os

import numpy as np
from PIL import Image


def load_data(n_from, n_to):
    root_folder = 'E:\\ml\\moles'
    img_folder = root_folder + '\\Images'
    desc_folder = root_folder + '\\Descriptions'

    m = n_to - n_from
    images = np.zeros((m, 750, 1000, 3))
    counter = 0
    for file in os.listdir(img_folder)[n_from:n_to]:
        img = Image.open(os.path.join(img_folder, file))
        images[counter] = np.array(img.resize((1000, 750)))
        counter += 1
        print(str(counter) + ': Loaded ' + str(file))
    print('Shape' + str(np.shape(images)))

    counter = 0
    labels = np.zeros((m,), dtype=np.int8)
    for file in os.listdir(desc_folder)[n_from:n_to]:
        desc = json.load(open(os.path.join(desc_folder, file)))
        if desc['meta']['clinical']['benign_malignant'] == 'benign':
            labels[counter] = 1
        counter += 1
        if counter % 100 == 0:
            print('Loaded ' + str(counter) + ' labels')
    print('All labels loaded')

    return images / 225., np.copy(labels)
