import cv2
import numpy as np

class Im(object):
    def __init__(self, im_file):
        self.image = cv2.imread(im_file, cv2.IMREAD_COLOR)
        self.image2x = cv2.resize(self.image, (0,0), fx=2., fy=2.)
        self.image3x = cv2.resize(self.image, (0,0), fx=3., fy = 3.)
        self.image512 = self.im2chipsize(self.image)

    # def cut_chips(self):

    def im2chipsize(self,image):
        im_max_size = max(self.image.shape[:2])
        # print(imMaxSize)
        return cv2.resize(image, (0,0), fx = 513./im_max_size, fy = 513./im_max_size)
        # print(self.image2x.shape)
    def readNegProposal(self):
        return []
    def getGroundTruth(self):
        return []
    # def genPosChip(self):
        


    def genChipCandidate(self, w, h):
        # chips = [[]]
        #x,y top-left index
        x_inds = np.arange(-32, w-512, 32) + 32
        xlen = len(x_inds)
        y_inds = np.arange(-32, h-512, 32) + 32
        ylen = len(y_inds)
        x_inds = np.array([x_inds,] * ylen).flatten()
        y_inds = np.array([y_inds,] * xlen).flatten(order='F')
        w_inds = np.ones(len(x_inds))
        chips = np.vstack((x_inds, y_inds))
        print(w_inds)
        # y = np.
        # while(y + 512 < h):
        #     while(x + 512 < w):
        #         chips.appenwhile(y + 512 < h):
        #     while(x + 512 < w):
        #         chips.append([x,y,512,512])
        #         x += 32
        #     chips.append([x,y,w-x,512])
        #     x = 0
        #     y += 32
        # while(x + 512 < w):
        #     chips.append([x,y,512,512])
        #     x += 32d([x,y,512,512])
        #         x += 32
        #     chips.append([x,y,w-x,512])
        #     x = 0
        #     y += 32
        # while(x + 512 < w):
        #     chips.append([x,y,512,512])
        #     x += 32

    def BoxInsideChip(self,box,chip):
        # if(box[0])
        return True
    


image = Im('./aachen_000000_000019_leftImg8bit.png')
# print(image.image512.shape)
print(image.genChipCandidate(1000, 2000))
