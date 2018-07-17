import cv2
import numpy as np
import json
import os


class Im2Chip(object):
    def __init__(self, im_file, gt_list, im_path):
        self.imname = im_file
        self.impath = os.path.join(im_path, im_file)
        self.gt_list = gt_list
        self.rp_list = [[1, 1, 1, 1], [32, 32, 32, 32], [300, 300, 300, 300]]

        self.image = cv2.imread(self.impath, cv2.IMREAD_COLOR)
        self.image_chips_candidates = self.__genChipCandidate(self.image.shape)

        self.image2x = cv2.resize(
            self.image, (0, 0), fx=2., fy=2., interpolation=cv2.INTER_LINEAR)
        self.image2x_chips_candidates = self.__genChipCandidate(
            self.image2x.shape)
        self.image3x = cv2.resize(
            self.image, (0, 0), fx=3., fy=3., interpolation=cv2.INTER_LINEAR)
        self.image3x_chips_candidates = self.__genChipCandidate(
            self.image3x.shape)
        self.image512 = self.__im2ChipSize(self.image)
        self.image512_chips_candidates = self.__genChipCandidate(
            self.image512.shape)

    def genChipMultiScale(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        chips1, chips_gts1 = self.__genChip(
            self.image, self.image_chips_candidates, 1., [128, 320])
        chips2, chips_gts2 = self.__genChip(
            self.image2x, self.image2x_chips_candidates, 2, [80, 160])
        chips3, chips_gts3 = self.__genChip(
            self.image3x, self.image3x_chips_candidates, 3., [0, 100])
        chips4, chips_gts4 = self.__genChip(
            self.image512, self.image512_chips_candidates,
            512 / max(self.image.shape), [300, 100000])
        chips = chips1 + chips2 + chips3 + chips4
        chips_gts = chips_gts1 + chips_gts2 + chips_gts3 + chips_gts4
        print(chips_gts)
        gt_out = {}
        for i in range(len(chips)):
            origin_name = self.imname.split('.')[0]
            new_name = origin_name + str('1%02d' % i) + '.jpg'
            new_path = os.path.join(path, new_name)
            new_chip = np.array(chips[i])
            # TODO:
            # this is a error!!!
            # resize do not keep original shape and order
            # new_chip.resize((512, 512, 3))
            # new_chip = np.zeros((512,512,3)).astype(np.uint8)
            # new_chip[0:,0:slice_data.shape[1],:] += slice_data
            # cv2.imwrite(new_path, new_chip)
            # gt_out[new_name] = chips_gts[i]
            # for gt in chips_gts[i]:
            #     chip = gt[1:]
            #     chip = list(map(int, chip))
            #     cv2.rectangle(chips[i], (chip[0], chip[1]),
            #                   (chip[0] + chip[2], chip[1] + chip[3]),
            #                   (255, 0, 0), 2)
            #     cv2.imshow('image', np.array(chips[i]))
            #     cv2.waitKey(0)
        return gt_out

    def genTestImg(self, length, path, position_file):
        image_slice0, image_data_0 = self.__genTestImgSingleScale(
            self.image512, length, 0)
        image_slice1, image_data_1 = self.__genTestImgSingleScale(
            self.image, length, 1)
        image_slice2, image_data_2 = self.__genTestImgSingleScale(
            self.image2x, length, 2)
        image_slice3, image_data_3 = self.__genTestImgSingleScale(
            self.image3x, length, 3)
        image_slice = {
            **image_slice0,
            **image_slice1,
            **image_slice2,
            **image_slice3
        }
        image_data = {
            **image_data_0,
            **image_data_1,
            **image_data_2,
            **image_data_3
        }
        for im_name in image_slice:
            im_path = os.path.join(path, im_name)
            # cv2.imshow('name' , image_slice[im_name])
            cv2.imwrite(im_path, image_slice[im_name])
        # with open(os.path.join(path,self.imname), 'w') as outfile:
        #     json.dump(image_data, outfile)
        return image_data

    def __genTestImgSingleScale(self, image, length, scale):
        image_slice = {}
        image_info = {}
        [h, w] = image.shape[0:2]
        x_slice_num = int(w // length) + 1
        y_slice_num = int(h // length) + 1
        if not x_slice_num == 1:
            x_slice_num += 1
        if not y_slice_num == 1:
            y_slice_num += 1
        x_top_left_pos = np.linspace(
            0, w - length, x_slice_num, endpoint=True, dtype=int)
        y_top_left_pos = np.linspace(
            0, h - length, y_slice_num, endpoint=True, dtype=int)
        top_left_pos = [[x, y] for x in x_top_left_pos for y in y_top_left_pos]
        for i in range(len(top_left_pos)):
            slice_name = '%s_%d_%02d.jpg' % (self.imname.split('.')[0], scale,
                                             i)
            slice_data = np.array(
                image[top_left_pos[i][1]:top_left_pos[i][1] +
                      length, top_left_pos[i][0]:top_left_pos[i][0] + length])
            slice_reshape = np.zeros((length, length, 3)).astype(np.uint8)
            slice_reshape[0:slice_data.shape[0], 0:slice_data.shape[
                1], :] += slice_data
            image_slice[slice_name] = slice_reshape
            image_info[slice_name] = top_left_pos[i] + [scale]
        return image_slice, image_info

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

    def __countOverlap(self, contains):
        candidate_contains_size = []
        candidate_contains = []
        for i in range(len(contains)):
            match = set()
            for j in range(len(contains[0])):
                if contains[i][j] == True:
                    match.add(j)
            candidate_contains.append(match)
            candidate_contains_size.append(len(match))
        return candidate_contains_size, candidate_contains

    def __im2ChipSize(self, image):
        im_max_size = max(self.image.shape[:2])
        return cv2.resize(
            image, (0, 0),
            fx=512. / im_max_size,
            fy=512. / im_max_size,
            interpolation=cv2.INTER_LINEAR)

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
        candidate_contains_size = []
        candidate_contains = []
        for i in range(len(chip_candidates)):
            contain = set()
            for j in range(len(gt_list)):
                if self.__contain_single_box(chip_candidates[i], gt_list[j]):
                    contain.add(j)
                    contains[i][j] = True
                    if j in gt2candidates:
                        gt2candidates[j].add(i)
                    else:
                        gt2candidates[j] = set()
                        gt2candidates[j].add(i)
            candidate_contains.append(contain)
            candidate_contains_size.append(len(contain))
        return candidate_contains, candidate_contains_size, gt2candidates

    def __genChip(self, image, chip_candidates, scale, s_range):
        [box_min, box_max] = s_range
        gt_filtered = np.array([])
        if not len(self.gt_list) == 0:
            gt_filtered = np.array([
                s for s in self.gt_list if (s[3] >= box_min or s[4] >= box_min)
                and s[3] < box_max and s[4] < box_max
            ]).astype(float)
        if not len(gt_filtered) == 0:
            gt_filtered[:, 1:] *= scale
            # print(chip_candidates)
        chips_pos = self.__genPosChips(chip_candidates, gt_filtered)
        rp_filtered = np.array([
            s for s in self.rp_list if (s[2] >= box_min or s[3] >= box_min)
            and s[2] < box_max and s[3] < box_max
        ]).astype(float)
        rp_filtered *= scale
        chips_neg = self.__genNegChips(chip_candidates, chips_pos, rp_filtered,
                                       1, 2)
        chips_shape = chip_candidates[chips_neg + chips_pos].astype(int)
        # print(chips_pos_test)
        # print(chips_pos)
        chips, chips_gts = self.__genChipsGt(chips_shape, image, gt_filtered)
        # chips_gts = []
        return chips, chips_gts

    def __genPosChips(self, chip_candidates, gt_filtered):
        gt_boxes = gt_filtered[:, 1:] if len(gt_filtered) > 0 else []
        candidate_contains, candidate_contains_size, gt2candidates = self.__overlap(
            chip_candidates, gt_boxes)
        # print(candidate_contains_size)
        chips = []
        gt_checked = set()
        candidate_contains_max = np.argmax(candidate_contains_size)
        while not (candidate_contains_size[candidate_contains_max] == 0):
            chips.append(candidate_contains_max)
            # chip_gt = []
            gt_inside_list = list(candidate_contains[candidate_contains_max])
            for gt_inside_index in gt_inside_list:
                # delete from candidate contain list
                if gt_inside_index not in gt_checked:
                    for candidate_index in gt2candidates[gt_inside_index]:
                        candidate_contains_size[candidate_index] -= 1
                    gt_checked.add(gt_inside_index)
            candidate_contains_max = np.argmax(candidate_contains_size)
        return chips

    def __genNegChips(self, chip_candidates, chips_pos, rp_filtered, rpn_count,
                      n):
        candidate_contains, candidate_contains_size, rp2candidates = self.__overlap(
            chip_candidates, rp_filtered)
        checked_rp = set()
        for chosen_chip in chips_pos:
            for rp in candidate_contains[chosen_chip]:
                if rp not in checked_rp:
                    checked_rp.add(rp)
                    for candidate in rp2candidates[rp]:
                        candidate_contains_size[candidate] -= 1
        # print(candidate_contains_size)
        candidate_contains_size = np.array(candidate_contains_size)
        chip_neg = np.argwhere(
            candidate_contains_size >= rpn_count).flatten().tolist()
        np.random.shuffle(chip_neg)
        return chip_neg[0:n]

    def __genChipsGt(self, chips_shape, image, gt_filtered):
        gt_boxes = gt_filtered[:, 1:] if len(gt_filtered) > 0 else []
        # add to output gt
        # print(chips_shape)
        chips = [
            image[s[1]:(s[1] + s[3]), s[0]:(s[0] + s[2])] for s in chips_shape
        ]
        chip_contains, chip_contains_size, gt2chips = self.__overlap(
            chips_shape, gt_boxes)
        # print(chip_contains)
        chip_gts = []
        for i in range(len(chips_shape)):
            chip_gt = []
            for gt_index in chip_contains[i]:
                gt = gt_filtered[gt_index].copy()
                gt[1:3] -= chips_shape[i][0:2]
                chip_gt.append(gt)
            chip_gts.append(chip_gt)
        for i in range(len(chips)):
            chips[i] = chips[i].copy()
            for gt in chip_gts[i]:
                gt = gt.astype(int)
                cv2.rectangle(chips[i], (gt[1], gt[2]),
                              (gt[1] + gt[3], gt[2] + gt[4]), (255, 0, 0), 1)
            cv2.imshow('image', chips[i])
            cv2.waitKey(0)

        return chips, chip_gts