#
# This example creates an HDF5 file from an image directory
#

import argparse
import torch
import os
import sys
import progressbar
from random import shuffle
from skimage import io, transform
from torchvision import transforms, utils

import h5py
import numpy as np


parser = argparse.ArgumentParser(description='Creates an HDF5 file from an image directory')
parser.add_argument('--source', metavar='SOURCE_FOLDER_PATH', help='Source folder for images', default=".")
parser.add_argument('--nbBatches', metavar='N', type=int, help='Number of batches to store', default=6000)
parser.add_argument('--output', metavar='OUTPUT_FILE_NAME', help='Name of output file', default="dataset.h5")
parser.add_argument('--imagesSize', metavar='S', type=int, help='Size of the images patches', default=28)
parser.add_argument('--batchSize', metavar='B', type=int, help='Number of images per batch', default=128)
args = parser.parse_args()

# import 'dataloader' # -- This code can be moved in a different file, packing everything together for easy sharing
class DataLoader:
    def __init__(self, folder, imageSize):
        self.fileList = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and (f.endswith("jpg") or f.endswith("png"))]
        self.imageSize = imageSize
        self.cursor = 0        
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(30), transforms.Resize(512), transforms.RandomCrop(self.imageSize), transforms.ToTensor()])

    def isDone(self):
        return self.cursor >= self.datasetSize() - 1

    def reset(self):
        shuffle(self.fileList)
        self.cursor = 0

    def datasetSize(self):
        return len(self.fileList)

    def getBatch(self, batchSize, b):
        b.fill_(0)
        for i in range(batchSize):
            loaded = False
            while not loaded:
                filename = self.fileList[self.cursor]
                self.cursor = (self.cursor + 1) % self.datasetSize()
                try:
                    loaded = True
                    image = io.imread(filename)
                    if (len(image.shape) == 3 and image.shape[2] >= 3):
                        b[i].copy_(self.transform(image[:,:,0:3]))
                except Exception as e:
                    print >> sys.stderr, "Error loading file " + filename + " ["+ str(e) +"]"
                    loaded = False
                
        return b
# /import 'dataloader'


NB_RECORD_TO_STORE = args.nbBatches
IMAGE_SIZE = args.imagesSize
IMAGE_FOLDER = args.source
BATCH_SIZE = args.batchSize
OUTPUT_FILE = args.output

print("Creating " + OUTPUT_FILE + " file out of " + IMAGE_FOLDER + " (crop size : " + str(IMAGE_SIZE) + " || batch size : " + str(BATCH_SIZE) + " )")

file = h5py.File(OUTPUT_FILE,'w')
image_type = np.dtype([('Path', 'i'), ('Data', np.float32, (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))])

loader = DataLoader(IMAGE_FOLDER, IMAGE_SIZE)
batch = torch.Tensor(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
if loader.datasetSize() == 0:
    print("No images found in source folder [" + IMAGE_FOLDER + "], specify different source folder with --source")
    sys.exit()

with progressbar.ProgressBar(max_value=NB_RECORD_TO_STORE) as bar:
    for i in range(0, NB_RECORD_TO_STORE):
        dataset = file.create_dataset("Batch_" + str(i), (1,), image_type)
        batch = loader.getBatch(BATCH_SIZE , batch)
        if loader.isDone():
            loader.reset()
        if batch.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE):
            dataset[0] = (i, batch)
        else:
            print("Returned batch has wrong size ", batch.size())
        file.flush()
        bar.update(i)


file.close()


#
# 1 batch is 128 x 3 x 28 x 28
# ==> 128 * 3 * 28 * 28 * (16 bits) = 0.602112 megabytes
# I can fit about 40,000 batches in a 25GB file
# That leaves about 7Go of RAM free for others processes
#
# Turns out creating a 40000 records file will take a few days
# So cutting it to 6000
#
