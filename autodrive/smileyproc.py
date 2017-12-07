'''
Stop sign detector worker
'''
from workerutil import ControlCommand
from workerutil import get_img

import cv2
import numpy as np
import time
from collections import deque

SMILEY_PARAMS = 'resources/smiley_object_cascade.xml'

SMILEY_SIZE_THRESHOLD = 0.1

SMILEY_COUNT = 1

lastframes = deque(10*[0], 10)

def smiley_proc(img_req, img_q, cmd_q):
  '''The stop sign worker process.'''
  classifier = cv2.CascadeClassifier(SMILEY_PARAMS)

  while True:
    # get a frame, and save a copy of it for later
    fr = get_img(img_req, img_q)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    fr = cv2.resize(fr, (320, 240))

    now = time.time()

    # do the cascade classifier
    #signs = classifier.detectMultiScale(fr, 1.02, 10)
    signs = classifier.detectMultiScale(fr, 1.3, 5)

    isSign = False

    for x, y, w, h in signs:
      isSign = True
      if(w >= 60 and h >= 60):
        lastframes.append(1)
      else:
        lastframes.append(0) 

    if not isSign:
      lastframes.append(0)     

    if len(suggested_stop_times) >= SMILEY_COUNT and lastframes.count(1) >= 5: 
      cmd_q.put(ControlCommand('start'))
      cmd_q.put(ControlCommand('speed', 0))
      suggested_stop_times = []

    else:
      cmd_q.put(ControlCommand('stop'))
      suggested_stop_times = []    

