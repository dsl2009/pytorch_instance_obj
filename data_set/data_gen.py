import random
import numpy as np
import config
from dsl_data import aug_utils
from dsl_data import xair_guoshu, mianhua, bdd, voc, Lucai,BigLand
import math
from data_set.data_set_utils import *

def get_batch(batch_size,class_name, is_shuff = True,max_detect = 50, is_rcnn = False):
    if class_name == 'guoshu':
        data_set = xair_guoshu.Tree('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree',
                                    config.image_size)
    elif class_name == 'mianhua':
        data_set = mianhua.MianHua('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/mianhua/open',
                                   config.image_size)
    elif class_name == 'bdd':
        data_set = bdd.BDD(js_file='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json',
                           image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train',
                                   image_size=config.image_size, is_crop=False)
    elif class_name == 'bdd_crop':
        data_set = bdd.BDD(js_file='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json',
                           image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train',
                                   image_size=config.image_size, is_crop=True)
    elif class_name == 'voc':
        data_set = voc.VOCDetection(root='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit',
                                   image_size=config.image_size)
    elif class_name == 'lvcai':
        data_set = Lucai.Lucai(image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/round2', image_size=config.image_size, is_crop=False)

    length = data_set.len()
    idx = list(range(length))

    width_ratio = config.output_size[1] / config.image_size[1]
    height_ratio =  config.output_size[0] / config.image_size[0]

    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img, box, lab = data_set.pull_item(idx[index])
            except:
                index = index+1
                if index >= length:
                    index = 0
                continue

            if  img is None or len(lab) == 0 or len(lab)>100:
                index+=1
                if index >= length:
                    index = 0
                continue
            box = box*config.image_size[0]
            img = (img - [123.15, 115.90, 103.06]) / 255.0
            img = np.transpose(img, axes=(2, 0, 1))
            categories = 20

            tl_heatmaps = np.zeros((categories, config.output_size[0], config.output_size[1]), dtype=np.float32)
            br_heatmaps = np.zeros((categories, config.output_size[0], config.output_size[1]), dtype=np.float32)

            tl_regrs = np.zeros((max_detect, 2), dtype=np.float32)
            br_regrs = np.zeros((max_detect, 2), dtype=np.float32)

            tl_tags = np.zeros((max_detect,), dtype=np.int64)
            br_tags = np.zeros((max_detect,), dtype=np.int64)
            tag_masks = np.zeros((max_detect,), dtype=np.uint8)


            for ind, detection in enumerate(box):
                category = lab[ind]

                xtl, ytl = detection[0], detection[1]
                xbr, ybr = detection[2], detection[3]

                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)

                xtl, ytl, xbr, ybr =np.clip(np.asarray([fxtl, fytl, fxbr, fybr],dtype=np.int),a_min=0, a_max=127)


                if True:
                    width = detection[2] - detection[0]
                    height = detection[3] - detection[1]

                    width = math.ceil(width * width_ratio)
                    height = math.ceil(height * height_ratio)

                    if True:
                        radius = gaussian_radius((height, width), 0.7)
                        radius = max(0, int(radius))
                    else:
                        radius = 3

                    draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius)
                    draw_gaussian(br_heatmaps[category], [xbr, ybr], radius)
                else:
                    tl_heatmaps[category, ytl, xtl] = 1
                    br_heatmaps[category, ybr, xbr] = 1

                tl_regrs[ind, :] = [fxtl - xtl, fytl - ytl]
                br_regrs[ind, :] = [fxbr - xbr, fybr - ybr]

                tl_tags[ind] = ytl * config.output_size[1] + xtl
                br_tags[ind] = ybr * config.output_size[1] + xbr

            tag_masks[:box.shape[0]] = 1


            if b== 0:
                images = np.zeros((batch_size, 3, config.image_size[0], config.image_size[1]), dtype=np.float32)
                b_tl_heatmaps = np.zeros((batch_size,categories, config.output_size[0], config.output_size[1]), dtype=np.float32)
                b_br_heatmaps = np.zeros((batch_size,categories, config.output_size[0], config.output_size[1]), dtype=np.float32)

                b_tl_regrs = np.zeros((batch_size,max_detect, 2), dtype=np.float32)
                b_br_regrs = np.zeros((batch_size,max_detect, 2), dtype=np.float32)

                b_tl_tags = np.zeros((batch_size,max_detect,), dtype=np.int64)
                b_br_tags = np.zeros((batch_size,max_detect,), dtype=np.int64)
                b_tag_masks = np.zeros((batch_size,max_detect,), dtype=np.uint8)



                images[b] = img
                b_tl_heatmaps[b] = tl_heatmaps
                b_br_heatmaps[b] = br_heatmaps
                b_tl_regrs[b] = tl_regrs
                b_br_regrs[b] = br_regrs
                b_tl_tags[b] = tl_tags
                b_br_tags[b] = br_tags

                b_tag_masks[b] = tag_masks
                index=index+1
                b=b+1
            else:
                images[b] = img
                b_tl_heatmaps[b] = tl_heatmaps
                b_br_heatmaps[b] = br_heatmaps
                b_tl_regrs[b] = tl_regrs
                b_br_regrs[b] = br_regrs
                b_tl_tags[b] = tl_tags
                b_br_tags[b] = br_tags
                b_tag_masks[b] = tag_masks

                index = index + 1
                b = b + 1
            if b>=batch_size:
                yield [images,b_tl_tags,b_br_tags, b_tl_heatmaps, b_br_heatmaps, b_tag_masks, b_tl_regrs, b_br_regrs]
                b = 0
            if index>= length:
                index = 0

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    gen = get_batch(batch_size=1, class_name='voc', max_detect=100)
    for x in range(20):
        _, tagtl, tagbr, tlheatmaps, bheatmaps, mask, _, _ = next(gen)
        print(tagtl,tagbr)
        plt.subplot(121)
        plt.imshow(np.sum(tlheatmaps[0, :, :, :], axis=0))
        plt.subplot(122)
        plt.imshow(np.sum(bheatmaps[0, :, :, :], axis=0))
        plt.show()




