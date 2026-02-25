import math
import re
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
import sys
from glob import glob

# MAP-THOR: add PDL repo root to sys.path so AI2Thor.Tasks can be imported.
# Assumes CWD is the LaMMA-P directory (AI2Thor/baselines/LaMMA-P within the PDL repo).
_pdl_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
if _pdl_root not in sys.path:
    sys.path.insert(0, _pdl_root)

try:
    from AI2Thor.Tasks.get_scene_init import get_scene_initializer
except ImportError as _e:
    print(f"[MAP-THOR] WARNING: could not import get_scene_initializer ({_e}). "
          "Scene preinit and checker will be disabled.")
    def get_scene_initializer(task, scene, ignore_applicable=False):
        return None, None

def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video():
    frame_rate = 5
    if shutil.which("ffmpeg") is None:
        print("[Video] ffmpeg not found; skipping video generation.")
        return
    # _exec_dir is injected by aithor_connect.py at module level; fall back to cwd
    _vid_dir = globals().get('_exec_dir', os.path.abspath(os.getcwd()))
    cur_path = _vid_dir + "/*/"
    for imgs_folder in glob(cur_path, recursive = False):
        view = imgs_folder.split('/')[-2]
        if not os.path.isdir(imgs_folder):
            print("The input path: {} you specified does not exist.".format(imgs_folder))
        else:
            command_set = ['ffmpeg', '-i',
                                '{}/img_%05d.png'.format(imgs_folder),
                                '-framerate', str(frame_rate),
                                '-pix_fmt', 'yuv420p',
                                '{}/video_{}.mp4'.format(_vid_dir, view)]
            subprocess.call(command_set)
        

