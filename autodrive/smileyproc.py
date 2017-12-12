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

COUNT_THRESHOLD = 2
SIZE_THRESHOLD = 100

def smiley_proc(img_req, img_q, cmd_q):
  '''The stop sign worker process.'''
  classifier = cv2.CascadeClassifier(SMILEY_PARAMS)

  lastframes = deque([0]*7, 7)

  while True:
    # get a frame, and save a copy of it for later
    fr = get_img(img_req, img_q)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    fr = cv2.resize(fr, (320, 240))

    now = time.time()

    # do the cascade classifier
    signs = classifier.detectMultiScale(fr, 1.05, 5)

    isSign = False

    for x, y, w, h in signs:
      isSign = True
      if w >= COUNT_THRESHOLD and h >= COUNT_THRESHOLD:
        lastframes.append(1)
        break

    if not isSign:
      lastframes.append(0)     

    if lastframes.count(1) >= COUNT_THRESHOLD:
      cmd_q.put(ControlCommand('start'))
      cmd_q.put(ControlCommand('speed', 0))

    else:
      cmd_q.put(ControlCommand('stop'))

