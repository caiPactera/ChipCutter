import json
from creat_chip import Im2Chip
import numpy as np
import os
import pickle, pprint
import pandas
# PATH = 'instances_train2014.json'
VAL_PATH = 'train_annotations.txt'
IMG_PATH = './train'
POSITION_PATH = 'position.txt'
PROPOSAL_PATH = 'rpn_proposals.pkl'
OUT_PATH = './output_images'

# rp = pandas.read_pickle(PROPOSAL_PATH)

pkl_file = pandas.read_pickle(PROPOSAL_PATH)
rp_dict = {}
for i in range(len(pkl_file['ids'])):
    rp_dict['img%05d.jpg' % int(pkl_file['ids'][i])] = pkl_file['boxes'][i]



# read annotation file into a dictionary
# file name as key
with open(VAL_PATH) as f:
    anno_lines = f.readlines()
gt_dict = {}

for l in anno_lines:
    l_contents = l.split()
    l_boxes = l_contents[1:]
    gt_dict[l_contents[0]] = np.array(
        [list(map(int, l_boxes[i:i + 5])) for i in range(0, len(l_boxes), 5)])
# iter by file name
f_out = open('annotation.txt', 'w')
img_info_total = {}
# for key in gt_dict:
for key in os.listdir(IMG_PATH):
    print(key)
    # if not os.path.isfile('train/%s' % key):
    #     continue
    # im_path = os.path.join(IMG_PATH,)
    # image_cutter = Im2Chip(key, gt_dict[key], IMG_PATH)

    image_cutter = Im2Chip(key, gt_dict[key], rp_dict[key], IMG_PATH)
    gts = image_cutter.genChipMultiScale(OUT_PATH)
    # img_info = image_cutter.genTestImg(1100, 'val_cut', 'position')
    # img_info_total.update(img_info)
# with open(POSITION_PATH, 'w') as position_file:
#     for key in img_info_total:
#         position_file.write(key)
#         position_file.write(' %d %d %d %d %d\n' %
#                             (img_info_total[key][0], img_info_total[key][1],
#                              1100, 1100, img_info_total[key][2]))
    for key in gts:
        f_out.write(key)
        for box in gts[key]:
            f_out.write(' ')
            f_out.write('%d %f %f %f %f' %tuple(box))
        f_out.write('\n')