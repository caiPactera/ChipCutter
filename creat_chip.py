import cv2
import numpy as np

class Im2Chip(object):
    def __init__(self, im_file):
        self.image = cv2.imread(im_file, cv2.IMREAD_COLOR)
        self.gt = {}



        self.image2x = cv2.resize(self.image, (0,0), fx=2., fy=2.)
        self.image2x_chips_candidates = self.genChipCandidate(self.image2x.shape)
        self.image3x = cv2.resize(self.image, (0,0), fx=3., fy = 3.)
        self.image3x_chips_candidates = self.genChipCandidate(self.image3x.shape)
        self.image512 = self.im2ChipSize(self.image)
        self.image512_chips_candidates = self.genChipCandidate(self.image512.shape)
        
        

    # def cut_chips(self):

    def im2ChipSize(self,image):
        im_max_size = max(self.image.shape[:2])
        return cv2.resize(image, (0,0), fx = 513./im_max_size, fy = 513./im_max_size)

    def readGroundTruth(self, path):
        



    def overlap(self, box1, box2):
        if box1[0] <= box2[0] and box1[1] <= box2[1] and (box1[0] + box1[2] >= box2[0] + box2[2]) and (box1[1] + box1[3] >= box2[1] + box2[3]):
            return True
        else:
            return False
        


    def genChipCandidate(self, shape):
        # chips = [[]]
        #x,y top-left index
        w = shape[0]
        h = shape[1]
        x_inds = np.arange(-32, w-512, 32) + 32
        xlen = len(x_inds)
        y_inds = np.arange(-32, h-512, 32) + 32
        ylen = len(y_inds)
        x_inds = np.array([x_inds,] * ylen).flatten()
        y_inds = np.array([y_inds,] * xlen).flatten(order='F')
        w_inds = np.ones(len(x_inds)) * 512
        h_inds = np.ones(len(y_inds)) * 512
        chips = np.vstack((x_inds, y_inds, w_inds, h_inds)).transpose()
        for chip in chips:
            if chip[0] + 512 > w:
                chip[2] = w - chip[0]
            if chip[1] + 512 > h:
                chip[3] = h - chip[1]
        # chips = chips.reshape(x_inds, y_inds, 4)
        return chips

    def BoxInsideChip(self,box,chip):
        # if(box[0])
        return True
    


image = Im2Chip('./aachen_000000_000019_leftImg8bit.png')
# print(image.image512.shape)
print(image.genChipCandidate([1000, 2000]))
