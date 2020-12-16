import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

import cv2
from scipy import ndimage

import time
import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


# TODO: merge this method with the one in BaseDetector
def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))

def get_box_cuda(cur_mask):
    pass

def get_box(mask):
    # if x_c < 0 or y_c < 0 or x_c == mask.shape[1] or y_c == mask.shape[0] or mask[x_c][y_c] != 1:
    #     return 0
    # for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
    #     next_i, next_j = x_c + di, y_c + dj
    #     get_box(mask, next_i, next_j)
    #     rect_x.append(next_i)
    #     rect_y.append(next_j)
    # return
    x = []
    y = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 0:
                y.append(i)
                x.append(j)
    x1 = min(x)
    y1 = min(y)
    x2 = max(x)
    y2 = max(y)
    print(x1,y1,x2,y2)
    return [x1,y1,x2,y2]

def show_result_ins(img,
                    result,
                    class_names,
                    score_thr=0.3,
                    sort_by_density=False,
                    out_file=None):
    """Visualize the instance segmentation results on the image.
    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]
        label_text = class_names[cur_cate]
        #label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        # sum_x = np.sum(cur_mask, axis=0)
        # x = np.where(sum_x > 0.5)[0]
        # sum_y = np.sum(cur_mask, axis=1)
        # y = np.where(sum_y > 0.5)[0]
        # x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        # img_show = cv2.rectangle(img_show, (x0,y0), (x1,y1), (0,255,0), 2)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
    if out_file is None:
        return img
    else:
        mmcv.imwrite(img_show, out_file)

def result_ins_box(img,
                    result,
                    class_names,
                    score_thr=0.3,
                    sort_by_density=False,
                    out_file=None):
    """Visualize the instance segmentation results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    person_inds = cate_label == 0
    seg_label = seg_label[person_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[person_inds]
    cate_score = cate_score[person_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    return_boxs = []
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]
        label_text = class_names[cur_cate]
        #label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        # prev_time = time.time()
        py,px = ndimage.measurements.find_objects(cur_mask)[0]
        box = [px.start,py.start,px.stop,py.stop]
        # print(py)
        # print(px)
        # print(px.start)
        # print(px.stop)
        # logger.info('ndimage process frame time:'+str(time.time()-prev_time))
        # global rect_x
        # global rect_y
        # mmcv.imwrite(cur_mask, 'cur_mask.jpg')
        # get_box_cuda(cur_mask)
        # prev_time = time.time()
        # box = get_box(cur_mask)
        # logger.info('process frame time:'+str(time.time()-prev_time))
        # x1 = min(rect_x)
        # y1 = min(rect_y)
        # x2 = max(rect_x)
        # y2 = max(rect_y)
        # rect_x=[]
        # rect_y=[]
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        img_show = cv2.rectangle(img_show, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
        # img_show = cv2.rectangle(img_show, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
        mmcv.imwrite(img_show, out_file)
        x = int(box[0])  
        y = int(box[1])  
        bw = int(box[2]-box[0])
        bh = int(box[3]-box[1])
        if x < 0 :
            bw = bw + x
            x = 0
        if y < 0 :
            bh = bh + y
            y = 0 
        box = [x,y,bw,bh]
        return_boxs.append([x,y,bw,bh])
    return return_boxs
    # if out_file is None:
    #     return img
    # else:
    #     mmcv.imwrite(img_show, out_file)

def result_ins_box_mask(img,
                    result,
                    class_names,
                    score_thr=0.3,
                    sort_by_density=False,
                    out_file=None):
    """Visualize the instance segmentation results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    # print(type(seg_label))
    # print(len(seg_label))
    # print(len(seg_label[0]))
    # print(len(seg_label[0][0]))
    # print(cur_result[1])
    cate_t = torch.where(cur_result[1] == 0)
    # print(cur_result[2][cate_t])
    vis_inds = torch.where(cur_result[2][cate_t] > score_thr)
    seg_label = seg_label[vis_inds]
    prev_time_f = time.time()
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    logger.info('front process frame time:'+str(time.time()-prev_time_f))
    num_mask = seg_label.shape[0]
    # cate_label = cur_result[1]
    # cate_label = cate_label.cpu().numpy()
    # score = cur_result[2].cpu().numpy()
    

    # vis_inds = score > score_thr
    # seg_label = seg_label[vis_inds]
    # num_mask = seg_label.shape[0]
    # cate_label = cate_label[vis_inds]
    # cate_score = score[vis_inds]
    
    # person_inds = cate_label == 0
    # seg_label = seg_label[person_inds]
    # num_mask = seg_label.shape[0]
    # cate_label = cate_label[person_inds]
    # cate_score = cate_score[person_inds]
    
    # if sort_by_density:
    #     mask_density = []
    #     for idx in range(num_mask):
    #         cur_mask = seg_label[idx, :, :]
    #         cur_mask = mmcv.imresize(cur_mask, (w, h))
    #         cur_mask = (cur_mask > 0.5).astype(np.int32)
    #         mask_density.append(cur_mask.sum())
    #     orders = np.argsort(mask_density)
    #     seg_label = seg_label[orders]
    #     cate_label = cate_label[orders]
    #     cate_score = cate_score[orders]
    
    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    # prev_time_f = time.time()
    # seg_label = seg_label.cpu().numpy().astype(np.uint8)
    # logger.info('front process frame time:'+str(time.time()-prev_time_f))
    return_boxs_mask = []
    prev_time_p = time.time()
    for idx in range(num_mask):
        prev_time = time.time()
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        logger.info('imresize process frame time:'+str(time.time()-prev_time))
        prev_time = time.time()
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        # cur_cate = cate_label[idx]
        # cur_score = cate_score[idx]
        # label_text = class_names[cur_cate]
        logger.info('cur_mask_bool process frame time:'+str(time.time()-prev_time))
        #label_text += '|{:.02f}'.format(cur_score)
        # center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        prev_time = time.time()
        # sum_x = np.sum(cur_mask, axis=0)
        # x = np.where(sum_x > 0.5)[0]
        # sum_y = np.sum(cur_mask, axis=1)
        # y = np.where(sum_y > 0.5)[0]
        cur_mask_tensor = torch.from_numpy(cur_mask)
        sum_x = torch.sum(cur_mask_tensor, dim=0)
        x = torch.where(sum_x > 0.5)[0]
        sum_y = torch.sum(cur_mask_tensor, dim=1)
        y = torch.where(sum_y > 0.5)[0]
        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        # py,px = ndimage.measurements.find_objects(cur_mask)[0]
        # box = [px.start,py.start,px.stop,py.stop]
        # box_mask = cur_mask[py.start:py.stop,px.start:px.stop]
        # box_img = img[py.start:py.stop,px.start:px.stop]
        logger.info('torch process frame time:'+str(time.time()-prev_time))
        box = [x0,y0,x1,y1]
        box_mask = cur_mask[y0:y1,x0:x1]
        box_img = img[y0:y1,x0:x1]
        ret,box_mask_bin = cv2.threshold(box_mask,0,255,cv2.THRESH_BINARY)
        # mmcv.imwrite(box_mask, 'box_mask.jpg')
        # mmcv.imwrite(box_img, 'box_img.jpg')
        # mmcv.imwrite(cur_mask, 'cur_mask.jpg')
        # mmcv.imwrite(box_mask_bin, 'box_mask_bin.jpg')
        x = int(box[0])  
        y = int(box[1])  
        bw = int(box[2]-box[0])
        bh = int(box[3]-box[1])
        if x < 0 :
            bw = bw + x
            x = 0
        if y < 0 :
            bh = bh + y
            y = 0 
        box = [x,y,bw,bh]
        return_boxs_mask.append([box,box_img,box_mask_bin])
        # mmcv.imwrite(img_show, out_file)
    logger.info('np process frame time:'+str(time.time()-prev_time_p))
    return return_boxs_mask,img_show