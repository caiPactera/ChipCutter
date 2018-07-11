import cv2
import numpy as np
import json
import os


class Im2Chip(object):
    def __init__(self, im_file, gt_list):
        self.imname = im_file
        self.image = cv2.imread(im_file, cv2.IMREAD_COLOR)
        self.gt_list = gt_list

        self.image2x = cv2.resize(self.image, (0, 0), fx=2., fy=2.)
        self.image2x_chips_candidates = self.__genChipCandidate(
            self.image2x.shape)
        self.image3x = cv2.resize(self.image, (0, 0), fx=3., fy=3.)
        self.image3x_chips_candidates = self.__genChipCandidate(
            self.image3x.shape)
        self.image512 = self.__im2ChipSize(self.image)
        self.image512_chips_candidates = self.__genChipCandidate(
            self.image512.shape)

    def __im2ChipSize(self, image):
        im_max_size = max(self.image.shape[:2])
        return cv2.resize(
            image, (0, 0), fx=512. / im_max_size, fy=512. / im_max_size)

    def __contain_single_box(self, chip, box):
        if chip[0] <= box[0] and chip[1] <= box[1] and (
                chip[0] + chip[2] >= box[0] + box[2]) and (chip[1] + chip[3] >=
                                                           box[1] + box[3]):
            return True
        else:
            return False

    def __overlap(self, chip_candidates, gt_list):
        contains = [
            x[:] for x in [[False] * len(gt_list)] * len(chip_candidates)
        ]
        gt2candidates = {}
        for i in range(len(chip_candidates)):
            for j in range(len(gt_list)):
                if self.__contain_single_box(chip_candidates[i],
                                             gt_list[j][1:]):
                    contains[i][j] = True
                    if j in gt2candidates:
                        gt2candidates[j].add(i)
                    else:
                        gt2candidates[j] = set()
                        gt2candidates[j].add(i)
        return contains, gt2candidates

    def __genChip(self, image, chip_candidates, scale, s_range):
        box_min = s_range[0]
        box_max = s_range[1]
        print(box_min)
        # print()

        gt_filtered = np.array([
            s for s in self.gt_list if (s[2] > box_min or s[3] > box_min)
            and s[2] < box_max and s[3] < box_max
        ]).astype(float)
        if not len(gt_filtered) == 0:
            gt_filtered[:, 1:] *= scale
        contains, gt2candidates = self.__overlap(chip_candidates, gt_filtered)
        candidates_contains_size = []
        candidates_contains = []
        for i in range(len(chip_candidates)):
            match = set()
            for j in range(len(gt_filtered)):
                if contains[i][j] == True:
                    match.add(j)
            candidates_contains.append(match)
            candidates_contains_size.append(len(match))
        chips_gts = []
        chips = []
        candidates_contains_max = np.argmax(candidates_contains_size)
        while not candidates_contains_size[candidates_contains_max] == 0:
            imcrop_box = chip_candidates[candidates_contains_max].astype(int)
            im_crop = image[imcrop_box[1]:(
                imcrop_box[1] + imcrop_box[3]), imcrop_box[0]:(
                    imcrop_box[0] + imcrop_box[2])]
            chips.append(im_crop)
            chip_gt = []
            gt_inside_list = list(candidates_contains[candidates_contains_max])
            for gt_inside_index in gt_inside_list:
                # add to output gt
                gt_scaled = gt_filtered[gt_inside_index]
                gt_scaled[1:3] -= chip_candidates[candidates_contains_max, 0:2]
                # print(chip_candidates[candidates_contains_max, 0:2])
                # print(gtoverlap_scaled)
                chip_gt.append(gt_scaled)

                # delete from candidate contain list
                for candidate_index in gt2candidates[gt_inside_index]:
                    # print(candidate_index)
                    candidates_contains_size[candidate_index] -= 1
                    candidates_contains[candidate_index].remove(
                        gt_inside_index)
            chips_gts.append(chip_gt)

            # for gt in chip_gt:
            #     chip = gt[1:]
            #     chip = list(map(int, chip))
            #     cv2.rectangle(im_crop, (chip[0], chip[1]),
            #                   (chip[0] + chip[2], chip[1] + chip[3]),
            #                   (255, 0, 0), 2)
            #     cv2.imshow('image', im_crop)
            #     cv2.waitKey(0)
            # TODO
            # crop gt on chip edge
            # for gt_index in self.gt_list:
            # print(gt_index)
            # print(chips_gts)

            candidates_contains_max = np.argmax(candidates_contains_size)

        return chips, chips_gts

    def genChipMultiScale(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        chips1, chips_gts1 = self.__genChip(
            self.image2x, self.image2x_chips_candidates, 512/max(self.image.shape), [32, 150])
        chips2, chips_gts2 = self.__genChip(
            self.image2x, self.image2x_chips_candidates, 2., [32, 150])
        chips3, chips_gts3 = self.__genChip(
            self.image3x, self.image3x_chips_candidates, 3., [120, 100000]
        )
        chips = chips1 + chips2 + chips3
        chips_gts = chips_gts1 + chips_gts2 + chips_gts3
        gt_out = {}
        for i in range(len(chips)):
            origin_name = self.imname.split('.')[0]
            new_name = origin_name + str('2%03d' % i) + '.jpg'
            new_path = os.path.join(path, new_name)
            new_chip = np.array(chips[i])
            new_chip.resize((512, 512, 3))
            cv2.imwrite(new_path, new_chip)
            gt_out[new_name] = chips_gts[i]
        # for gt in chips_gts[i]:
        #     chip = gt[1:]
        #     chip = list(map(int, chip))
        #     cv2.rectangle(chips[i], (chip[0], chip[1]),
        #                   (chip[0] + chip[2], chip[1] + chip[3]),
        #                   (255, 0, 0), 2)
        #     cv2.imshow('image', np.array(chips[i]))
        #     cv2.waitKey(0)
        return gt_out

    def __genChipCandidate(self, shape):
        # cv2 have revised order of shape
        h = shape[0]
        w = shape[1]
        x_inds = np.arange(0, max(w - 480, 32), 32)
        xlen = len(x_inds)
        y_inds = np.arange(0, max(h - 480, 32), 32)
        ylen = len(y_inds)
        x_inds = np.array([
            x_inds,
        ] * ylen).flatten()
        y_inds = np.array([
            y_inds,
        ] * xlen).flatten(order='F')
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
