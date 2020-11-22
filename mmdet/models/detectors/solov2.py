from .single_stage_ins import SingleStageInsDetector
# from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class SOLOv2(SingleStageInsDetector):
# class SOLOv2(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLOv2, self).__init__(backbone, neck, bbox_head, mask_feat_head, train_cfg,
                                   test_cfg, pretrained)
        # super(SOLOv2, self).__init__(backbone, neck, bbox_head, train_cfg,
        #                            test_cfg, pretrained)
