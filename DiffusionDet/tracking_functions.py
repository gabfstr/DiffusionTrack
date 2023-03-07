import motmetrics as mm
import numpy as np
import cv2 as cv
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import ipywidgets as widgets
def motMetricsEnhancedCalculator(gtSource, tSource, threshold=0.5, iou=0.5):
  # import required packages

  
  # load ground truth
  gt = np.loadtxt(gtSource, delimiter=',')

  # Filter non human boxes
  fltr= np.asarray([1,2,7])
  gt = gt[np.in1d(gt[:,7],fltr)]

  # load tracking output
  t = np.loadtxt(tSource, delimiter=',')
  t = t[t[:,6]>threshold]

  # Create an accumulator that will be updated during each frame
  acc = mm.MOTAccumulator(auto_id=True)

  # Max frame number maybe different for gt and t files
  for frame in range(int(gt[:,0].max())):
    frame += 1 # detection and frame numbers begin at 1

    # select id, x, y, width, height for current frame
    # required format for distance calculation is X, Y, Width, Height \
    # We already have this format
    gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
    t_dets = t[t[:,0]==frame,1:6] # select all detections in t
    
    C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=iou) # format: gt, t

    # Call update once for per frame.
    # format: gt object ids, t object ids, distance
    acc.update(gt_dets[:,0].astype('int').tolist(), \
              t_dets[:,0].astype('int').tolist(), C)

  mh = mm.metrics.create()

  summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ], \
                      name='acc')

  strsummary = mm.io.render_summary(
      summary,
      formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
              }
  )
  print(strsummary)
  return summary
  
  
def tracker_test(img_path, tracker, path_det_res, vid_output_path='vid_04.mp4', fps=30, res_output_path='',thresh=0.5,plot_results=False, write_ids=False):
    import glob
    import numpy as np
    frame_nb = 1
    img_array = []
    track_array =[]
    det_df = np.loadtxt(path_det_res, delimiter=',')
    det_df = det_df[det_df[:,6]>thresh]
    for filename in sorted(glob.glob(img_path+"img1/*.jpg")):
        image = cv.imread(filename)
        updated_image = image.copy()

        # NOTE: 
    # * `detection_bboxes` are numpy.ndarray of shape (n, 4) with each row containing (bb_left, bb_top, bb_width, bb_height)
    # * `detection_confidences` are numpy.ndarray of shape (n,);
    # * `detection_class_ids` are numpy.ndarray of shape (n,).

        #bboxes, confidences, class_ids = model.detect(image)
        bboxes = det_df[det_df[:,0]==frame_nb,2:6]
        confidences = det_df[det_df[:,0]==frame_nb,6]
        class_ids = det_df[det_df[:,0]==frame_nb,1]
        
        tracks = tracker.update(bboxes, confidences, class_ids)
        for t in tracks:
          track_array.append(t)
        #updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            clr = (0, 255,0) #[int(c) for c in self.bbox_colors[cid]]
            cv.rectangle(updated_image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])), clr, 2)
            # label = "{}:{:.4f}".format('Human', conf)
            # (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # y_label = max(bb[1], label_height)
            # cv.rectangle(updated_image, (int(bb[0]), int(y_label - label_height)), (int(bb[0] + label_width), int(y_label + baseLine)),
            #              (255, 255, 255), cv.FILLED)
            # cv.putText(updated_image, label, (int(bb[0]), int(y_label)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        updated_image = draw_tracks(updated_image, tracks)
        height, width, layers = image.shape
        size = (width,height)
        img_array.append(updated_image)
        frame_nb+=1
        # if write_ids:
        #   with open(res_output_path,"w") as output_file :
    out = cv.VideoWriter(vid_output_path,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
    if plot_results:
      for i in range(len(img_array)):
        out.write(img_array[i])
      out.release()
      print("Video made successfully :",vid_output_path)

    if write_ids:
      p ='/content/drive/MyDrive/OFFICIAL_recvis_proj/TrackEval/data/trackers/mot_challenge/MOT17-train/B_SORT/MOT17-'+img_path[-9:-7]+'-FRCNN.txt'
      print(p)
      with open(p,'w') as out_file:
        for row in track_array:
          out_file.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},-1,-1,-1\n')
    return track_array