import json
from creat_chip import Im2Chip
import numpy as np
import os
# PATH = 'instances_train2014.json'
VAL_PATH = 'train_annotations.txt'
IMG_PATH = 'train'

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
for key in gt_dict:
    print(key)
    # if not key == 'img00006.jpg':
    #     continue
    if not os.path.isfile('train/%s' % key):
        continue
    # im_path = os.path.join(IMG_PATH,)
    image_cutter = Im2Chip(key, gt_dict[key], IMG_PATH)
    # image_cutter.genChip(image_cutter.image3x, image_cutter.image3x_chips_candidates, 3, [0,512])
    # image_cutter.genChip(image_cutter.image512, image_cutter.image512_chips_candidates, 512/max(image_cutter.image.shape), [0,512])
    gts = image_cutter.genChipMultiScale('output_images')
    for key in gts:
        f_out.write(key)
        for box in gts[key]:
            f_out.write(' ')
            f_out.write('%d %f %f %f %f' %(box[0], box[1], box[2], box[3], box[4]))
        f_out.write('\n')