import argparse

import cv2
import torch
from scipy import ndimage


from mmdet.apis import inference_detector, init_detector, show_result, show_result_ins, result_ins_box,result_ins_box_mask
import mmcv

import numpy as np
import warnings
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools import generate_features as gfea
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

import time
import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config', default='configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.py', help='test config file path')
    parser.add_argument('--checkpoint', default='SOLOv2_LIGHT_448_R50_3x.pth', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    # parser.add_argument(
    #     '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--camera-id', default='MOT16-01.mp4', help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # deep_sort 
    # model_filename = 'model_data/mars-small128.pb'
    # encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    model_config = './model_data/deploy_mgcam.prototxt'
    model_data = './model_data/mgcam_iter_75000.caffemodel'
    encoder = gfea.create_box_encoder(model_config,model_data)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)

    # writeVideo_flag = True
    # if writeVideo_flag:
    # Define the codec and create VideoWriter object
    w = int(camera.get(3))
    h = int(camera.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('MOT16-01-solo-out-03.avi', fourcc, 15, (w, h))
        # list_file = open('detection.txt', 'w')
        # frame_index = -1 

    print('Press "Esc", "q" or "Q" to exit.')
    count = 0
    while True:
        ret_val, img = camera.read()
        if not ret_val:
            break
        prev_time = time.time()
        result = inference_detector(model, img)
        logger.info('process frame time:'+str(time.time()-prev_time))
        demo_path = './results/' + str(count) + '.jpg'
        # if ret_val:
        prev_time = time.time()
        boxs_masks,img_show = result_ins_box_mask(img, result, model.CLASSES, score_thr=0.3, out_file=demo_path) #0.25
        boxs_masks = np.array(boxs_masks)
        logger.info('postprocessing frame time:'+str(time.time()-prev_time))
        features = encoder(boxs_masks[:,1:])
    
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs_masks[:,0], features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(img_show, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(img_show,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            # save a frame
        out.write(img_show)
            # frame_index = frame_index + 1
            # list_file.write(str(frame_index)+' ')
            # if len(boxs) != 0:
            #     for i in range(0,len(boxs)):
            #         list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            # list_file.write('\n')
        # else:
        #     break
        count += 1
    camera.release()
    out.release()
        # list_file.close()

        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break

        # show_result(
        #     img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1)


if __name__ == '__main__':
    main()
