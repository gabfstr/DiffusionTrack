import os, sys
sys.path.insert(0, os.path.abspath('../detectron2/'))

import glob
import argparse
import multiprocessing as mp
import numpy as np
import os
import tqdm
import csv



# import detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image


from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer



# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #Set numclasses to 1
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES=1
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg




def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        default = "datasets/MOT17/train/*/img1"
    )

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        default = "output_mot_detection/det.txt"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Id of the class to be detected",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



    


if __name__ == "__main__" :
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)


    predictor = DefaultPredictor(cfg)

    output_path = args.output

    if len(args.input) >= 1:
        args.input = sorted(glob.glob((args.input)))
        assert args.input, "The input path(s) was not found"
    
    if args.output:
        assert len(args.output) >= 1, "Please specify a directory with args.output"
        out_filename = args.output


    if os.path.isfile(args.output):
        out_filename = os.path.split(out_filename)[0]
    
    logger.info("Output detections will be stored in {}".format(out_filename))


    for path in args.input:
        video_name = os.path.split(os.path.split(path)[0])[1]
        print("\n Scanning video {}".format(video_name),"\n")

        
        if args.confidence_threshold:
            output_path = os.path.join(out_filename,video_name + "_thresh_{}.txt".format(args.confidence_threshold))
        else:
            output_path = os.path.join(out_filename,video_name + "_det.txt")

        with open(output_path,"w") as output_file :

            frame_id=1
            inter_det_path = "inter_det_file/gab_session2/det.txt"
            print("Intermediate det path : ",inter_det_path)
            with open(inter_det_path,"w") as intermediate_det_file:
                        writer = csv.writer(intermediate_det_file)
                        writer.writerows([])
                        print("Intermediate detections file was reset")
            for img in tqdm.tqdm(sorted(glob.glob("{}/*.jpg".format(path)))):
                predictions = predictor(read_image(img,format="BGR"))
                #print("predicitons : ",predictions)             
                try :
                    scores = predictions['instances'].to('cpu').scores.numpy()
                    classes = predictions['instances'].to('cpu').pred_classes.numpy()
                    #print("classes of the predicitons : ",classes)
                    detection_list=[]
                    for j in range(len(scores)):
                        if (scores[j] > args.confidence_threshold) and (classes[j]==args.class_id) :
                        # class for human in lvis dataset : 792
                        # class for human in mot17 dataset : 0
                            box = predictions['instances'].to("cpu").pred_boxes.tensor[j].numpy()
                            x1 = int(box[0])
                            y1 = int(box[1])
                            x2 = int(box[2])
                            y2 = int(box[3])
                            output_file.write(f'{frame_id},-1,{x1},{y1},{x2-x1},{y2-y1},{scores[j]},-1,-1,-1\n')
                            detection_list.append([frame_id,-1,x1,y1,x2-x1,y2-y1,scores[j],-1,-1,-1])
                    with open(inter_det_path,"w") as intermediate_det_file:
                        writer = csv.writer(intermediate_det_file)
                        writer.writerows(detection_list)
                        #print("Intermediate detections written in inter_det_file/det.txt")
                except Exception as e :
                    print(e)
                frame_id+=1
        print("Results written in {}".format(output_path))
    
