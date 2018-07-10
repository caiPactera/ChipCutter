import cv2
import numpy as np
import json

class Im2Chip(object):
    def __init__(self, im_file, gt_list):
        self.image = cv2.imread(im_file, cv2.IMREAD_COLOR)
        self.gt_list = gt_list

        self.image2x = cv2.resize(self.image, (0,0), fx=2., fy=2.)
        self.image2x_chips_candidates = self.genChipCandidate(self.image2x.shape)
        self.image3x = cv2.resize(self.image, (0,0), fx=3., fy = 3.)
        self.image3x_chips_candidates = self.genChipCandidate(self.image3x.shape)
        self.image512 = self.im2ChipSize(self.image)
        self.image512_chips_candidates = self.genChipCandidate(self.image512.shape)
        
        self.overlap(self.image2x_chips_candidates)


    def im2ChipSize(self,image):
        im_max_size = max(self.image.shape[:2])
        return cv2.resize(image, (0,0), fx = 512./im_max_size, fy = 512./im_max_size)

    # def readGroundTruth(self, path):
    #     with open(path) as f:
    #         annotation_json = json.load(f)
    #     print(annotation_json)



    def contain_single_box(self, box1, box2):
        if box1[0] <= box2[0] and box1[1] <= box2[1] and (box1[0] + box1[2] >= box2[0] + box2[2]) and (box1[1] + box1[3] >= box2[1] + box2[3]):
            return True
        else:
            return False
    
    def overlap(self, chip_candidates, gt_list):
        contains = [x[:] for x in [[False] * len(gt_list)] * len(chip_candidates)]
        # print(contains)  
        for i in len(chip_candidates):
            for j in len(gt_list):
                if self.contain_single_box(chip_candidates[i], gt_list[j]):
                    contains[i][j]= True
        return contains
    
    def genChip(self, chip_candidates, scale):
        box_min = scale[0]
        box_max = scale[1]
        scale = scale[2]
        gt_filtered = [s for s in self.gt_list if (s[2] > box_min or s[3] > box_min) and s[2] < box_max and s[3] < box_max]
        contains = self.overlap(chip_candidates, gt_filtered)
        for i in range(len(chip_candidates)):
            match = []
            for j in range(len(gt_filtered)):
                if 

        


    def genChipCandidate(self, shape):
        w = shape[0]
        h = shape[1]
        x_inds = np.arange(0, max(w-480, 32), 32)
        xlen = len(x_inds)
        y_inds = np.arange(0, max(h-480, 32), 32)
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
        np.random.shuffle(chips)
        return chips

    def BoxInsideChip(self,box,chip):
        # if(box[0])
        return True
    


image = Im2Chip('img00005.jpg', [1])
# print(image.image512.shape)
# print(image.readGroundTruth('instances_minival2014.json'))
