import argparse
import os, glob
import sys
import tqdm
import logging
import os.path as osp
import multiprocessing as mp
import cv2
import numpy as np
import torch
import csv
import textwrap

sys.path.append('.')
sys.path.insert(0, os.path.abspath('./SMILEtrack/BoT-SORT/'))
sys.path.insert(0, os.path.abspath('./SMILEtrack/BoT-SORT/yolox/'))
sys.path.insert(0, os.path.abspath('./SMILEtrack/BoT-SORT/yolox/exps/example/mot/'))
sys.path.insert(0, os.path.abspath('./detectron2/'))
sys.path.insert(0, os.path.abspath('./DiffusionDet/'))

# sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../yolox'))



# import detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image


from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs

# import SMILEtrack utilities
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.visualize import plot_tracking

from tracker.tracking_utils.timer import Timer
from tracker.bot_sort import BoTSORT

from DiffusionTrack import setup_cfg, get_image_list, write_results, diffdet_detections, image_track
from DiffusionTrack import Predictor





IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("Multi-object tracking with DiffusionDet!")


    # Data
    parser.add_argument("path", default = "../../DiffusionDet/datasets/mot/" ,help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("-o", "--output-dir", default="output", type=str, help="desired output folder for experiment results")
    parser.add_argument("--save-det",help="If True, will store detections in an additional separate file.",default = True)


    # Experiment
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")


    # Detector
    parser.add_argument("--config-file",default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",metavar="FILE",help="path to config file",)
    parser.add_argument("--confidence-threshold","--det-thresh",type=float,default=0.5,help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",default=[],nargs=argparse.REMAINDER,)    

    # Parameters
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser




def main(args):

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    # run detection model
    detections = diffdet_detections(args)
    # run tracking model
    image_track(detections, args)




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = make_parser().parse_args()
    logger = setup_logger(name="DiffTrack")
    logger.info("Arguments: " + str(args))

    if len(args.path) >= 1:
        args.path = sorted(glob.glob((args.path)))
        assert args.path, "The input path(s) was not found"
    
    if args.output_dir:
        assert len(args.output_dir) >= 1, "Please specify a directory with args.output_dir"
        out_filename = args.output_dir

    if os.path.isfile(args.output_dir):
        out_filename = os.path.split(out_filename)[0]

    
    logger.info("Output tracked detections will be stored in {}".format(out_filename))
    

    data_path = args.path
    device = args.device

    
    mainTimer = Timer()
    mainTimer.tic()
    
    j=0
    for path in data_path:
        args.name = os.path.split(os.path.split(path)[0])[1]


        args.fps = 30
        args.device = device
        args.batch_size = 1
        args.trt = False
    
        if args.default_parameters:
        
            args.exp_file = r'./yolox/exps/example/mot/mot17_exp.py'
            args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'


            args.track_high_thresh = 0.6
            args.track_low_thresh = 0.1
            args.track_buffer = 30    
            args.new_track_thresh = args.track_high_thresh + 0.1
        j+=1

        print(textwrap.wrap('-'*150,width=150,max_lines=1)[0])
        ch="Processing video "+args.name + ' ({}/{})'.format(j,len(data_path))
        print('{:^120}'.format(ch))
        print(textwrap.wrap('-'*150,width=150,max_lines=1)[0])
        main(args)

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))
