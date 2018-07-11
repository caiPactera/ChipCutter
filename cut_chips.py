import json
from creat_chip import Im2Chip
import numpy as np
# PATH = 'instances_train2014.json'
VAL_PATH = 'train_annotations.txt'

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
for key in gt_dict:
    if not key == 'img00005.jpg':
        continue
    print(key)
    image_cutter = Im2Chip(key, gt_dict[key])
    # image_cutter.genChip(image_cutter.image3x, image_cutter.image3x_chips_candidates, 3, [0,512])
    # image_cutter.genChip(image_cutter.image512, image_cutter.image512_chips_candidates, 512/max(image_cutter.image.shape), [0,512])
    image_cutter.genChipMultiScale('output')
    break