import h5py
import numpy as np
import visdom
import time

vis = visdom.Visdom()
file = h5py.File('artsper.h5', 'r')
dataset = file["ARTSPER"]


images = dataset['Data']
print(images.shape)
for i in range(0, images.shape[0]):
    print(images[i].shape)
    print(images[i])
    vis.image(images[i][0], win="IMG")
    time.sleep(2)
file.close()
