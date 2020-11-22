import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result, show_result_ins

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

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)

    print('Press "Esc", "q" or "Q" to exit.')
    count = 0
    while True:
        ret_val, img = camera.read()
        prev_time = time.time()
        result = inference_detector(model, img)
        logger.info('process frame time:'+str(time.time()-prev_time))
        demo_path = './results/' + str(count) + '.jpg'
        if ret_val:
            prev_time = time.time()
            show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=demo_path)
            logger.info('postprocessing frame time:'+str(time.time()-prev_time))
        else:
            break
        count += 1
        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break

        # show_result(
        #     img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1)


if __name__ == '__main__':
    main()
