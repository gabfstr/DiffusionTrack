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
from detectron2.utils.logger import setup_logger


# import SMILEtrack utilities
from yolox.exp import get_exp

from tracker.tracking_utils.timer import Timer

from DiffusionTrack import diffdet_detections, image_track



IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
detectorTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("Multi-object tracking with DiffusionDet!")


    # Data
    parser.add_argument("path", default = "../../DiffusionDet/datasets/mot/" ,help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("-o", "--output-dir", default="output", type=str, help="desired output folder for experiment results")
    parser.add_argument("--save-det",help="If True, will store detections in an additional separate file.",default = True)
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")
    parser.add_argument("--det-folder","--detection-folder", default="", type=str, help="Set to folder with detections files in MOT17Det format to run track on existing detection file.")


    # Detector
    parser.add_argument("--config-file",default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",metavar="FILE",help="path to config file",)
    parser.add_argument("--confidence-threshold","--det-thresh",type=float,default=0.1,help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--class-id","--det-class",type=int,default=0,help="Id of the COCO class to be detected",)
    parser.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",default=[],nargs=argparse.REMAINDER,)

    # Parameters
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    

    return parser


#fill in other required args for the model with default values
def fill_required_args(args):
    
    #general
    args.benchmark='MOT17'
    args.eval='test'
    args.f=None
    args.c=None
    args.ablation = False

    #parameters
    args.experiment_name="DEMO"
    args.default_parameters=False
    args.conf=None
    args.nms=None
    args.tsize=None
    args.fp16=False
    args.fuse=False

    # tracking args
    args.track_high_thresh=0.6
    args.track_low_thresh=0.1
    args.new_track_thresh=0.7
    args.track_buffer=30
    args.match_thresh=0.8
    args.aspect_ratio_thresh=1.6
    args.min_box_area=10
    
    # CMC
    args.cmc_method="none"

    #ReID
    args.fast_reid_config=r"fast_reid/configs/MOT17/sbs_S50.yml"
    args.fast_reid_weights=r"pretrained/mot17_sbs_S50.pth"
    args.proximity_thresh=0.5
    args.appearance_thresh=0.25
    
    return args




def main(args):

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    exp=get_exp(args.exp_file, args.name)
    #exp.test_size = (736, 1920)
        
    exp.test_conf = max(0.001, args.track_low_thresh - 0.01)

    if len(args.det_folder)>=1 :
        timer.tic()
        #Load det files
        detections = np.loadtxt(args.det_folder,delimiter=',')
    
    else:
        timer.tic()
        # run detection model
        detections = diffdet_detections(args)
    # run tracking model
    image_track(exp, detections, args)




if __name__ == "__main__":
    
    #Setup process & args
    mp.set_start_method("spawn", force=True)
    args = make_parser().parse_args()
    
    #Additional required default args
    args=fill_required_args(args)

    #Setup logger
    logger = setup_logger(name="DiffTrack")
    logger.info("Arguments: " + str(args))

    #Test args
    if len(args.path) >= 1:
        args.path = sorted(glob.glob((args.path)))
        assert args.path, "The input path(s) was not found"
    
    if args.output_dir:
        assert len(args.output_dir) >= 1, "Please specify a directory with args.output_dir"
        out_filename = args.output_dir

    if os.path.isfile(args.output_dir):
        out_filename = os.path.split(out_filename)[0]

    if args.device == "gpu" or args.device == "cuda":
        args.device=torch.device("cuda")
    if args.device == "mps":
        args.device = torch.device("mps")
    else :
        args.device=torch.device("cpu")
    logger.info("Device : {}".format(args.device.type))
    
    
    logger.info("Output tracked detections will be stored in {}".format(out_filename))
    

    data_path = args.path
    device = args.device

    
    mainTimer = Timer()
    trackerTimer = Timer()
    timer = Timer()
    
    mainTimer.tic()
    
    #If using det files
    if len(args.det_folder)>=1 :
        det_files = sorted(glob.glob(args.det_folder + '/*'))
        logger.info("Loading detection files : "+str(det_files))
    

    #Iterating through image sequence
    j=0
    for path in data_path:
        args.name = os.path.split(os.path.split(path)[0])[1]
        args.path = path
        args.fps = 30
        args.device = device
        args.batch_size = 1
        args.trt = False
        args.mot20 = False
        
        
        args.exp_file = r'./yolox/exps/example/mot/mot17_exp.py'
        args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'


        args.track_high_thresh = 0.6
        args.track_low_thresh = 0.1
        args.track_buffer = 30    
        args.new_track_thresh = args.track_high_thresh + 0.1

        if len(args.det_folder)>=1 :
            args.det_folder=det_files[j]
        j+=1

        
        print(textwrap.wrap('-'*150,width=150,max_lines=1)[0])
        ch="Processing video "+args.name + ' ({}/{})'.format(j,len(data_path))
        print('{:^120}'.format(ch))
        print(textwrap.wrap('-'*150,width=150,max_lines=1)[0])
        main(args)

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
