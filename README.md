# DiffusionTrack
Finetuning &amp; extending DiffusionDet to video &amp; pedestrian multi-object-tracking


<img src="https://github.com/gabzouz37/DiffusionTrack/blob/main/DiffusionTrack.gif" width="100%"/>


Mutli-object tracking using diffusion object-detection (DiffusionDet finetuned for video) and similarity learning (adapted SMILEtrack on top)



1. Clone the repo 

```
git clone 'https://github.com/gabzouz37/DiffusionTrack/'
```

2. properly install detectron2
```
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

3. Install python requirements
```
!python -m pip install requirements.txt
```

-> Play with the Demo !

- run demo file on custom video with :
```
!python Demo.py 'PATH/TO/VIDEO' --configfile 'TESTFILE' ...
```
- Full collab Demo [Here](https://colab.research.google.com/drive/16ZBBvv3oj0DrnYTj3VjUkdVu7EPUI_w8#scrollTo=HrVkBM_aZBdP)
