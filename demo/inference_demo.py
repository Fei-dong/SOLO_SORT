from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv

import time
import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
config_file = 'configs/solov2/solov2_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth' # SOLOv2_R50_3x.pth
checkpoint_file = 'SOLOv2_R50_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = 'demo.jpg'
img = 'input.jpg'
prev_time = time.time()
result = inference_detector(model, img)
logger.info('process frame time:'+str(time.time()-prev_time))
# print(len(result[0]))
# print(len(result[0][0]))
# print(result[0][0][0])
# print(result[0][1])
# print(result[0][2])
# prev_time = time.time()
show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")
logger.info('postprocessing frame time:'+str(time.time()-prev_time))
