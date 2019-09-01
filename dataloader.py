import torch
import os
import sys
from random import shuffle
from skimage import io, transform
from torchvision import transforms, utils

class DataLoader:
    def __init__(self, folder, imageSize):
        self.fileList = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith("jpg")]
        self.imageSize = imageSize
        self.cursor = 0
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(self.imageSize), transforms.ToTensor()])

    def isDone(self):
        return self.cursor >= self.datasetSize() - 1

    def reset(self):
        shuffle(self.fileList)
        self.cursor = 0

    def datasetSize(self):
        return len(self.fileList)

    def getBatch(self, batchSize):
        batchSize = min(batchSize, self.datasetSize() - self.cursor)
        b = torch.Tensor(batchSize, 3, self.imageSize, self.imageSize)
        b.fill_(0)
        for i in range(batchSize):
            filename = self.fileList[self.cursor]
            try:
                image = io.imread(filename)
                if (len(image.shape) == 3 and image.shape[2] == 3):
                    b[i].copy_(self.transform(image))
            except Exception as e:
                print >> sys.stderr, "Error loading file " + filename + " ["+ str(e) +"]"
            self.cursor += 1
        return b
