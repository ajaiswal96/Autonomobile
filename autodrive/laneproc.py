'''
Lane detection worker
'''

from collections import deque
from detect import detect_lanes
from workerutil import ControlCommand
from workerutil import clamp
from workerutil import get_img

import cv2
import numpy as np
import time

# Scaling factors for each of the feedback system parameters
P, I, D = 40, 0, 10

# How many points of history to save for the D term
DERIV_HISTORY_LEN = 3

# How fast the car should go
SPEED_NORMAL = 3

# How fast the car should go if turning
SPEED_TURN = 4

def lane_proc(img_req, img_q, cmd_q):
  '''The lane guidance worker process.'''
  recorder = get_recorder('output.avi')

  # this process is always active
  cmd_q.put(ControlCommand('start'))
  cmd_q.put(ControlCommand('speed', SPEED_NORMAL))

  err_hist = deque()
  time_hist = deque()

  while True:
    # get the latest error
    fr = get_img(img_req, img_q)
    ll, rr, th = detect_lanes(fr, recorder=recorder)
    err = to_error(ll, rr, th)

    if err is None: continue

    now = time.time()
    err_hist.append(err)
    time_hist.append(now)

    # prune out the old errors
    #while now - time_hist[0] > DERIV_HISTORY_SEC:
    while len(time_hist) > DERIV_HISTORY_LEN:
      err_hist.popleft()
      time_hist.popleft()

    # calculate PID
    cur_err = err_hist[-1]
    total = np.sum(err_hist)
    slope = np.polyfit(time_hist, err_hist, 1)[0]

    p = P * -cur_err
    i = I * -total
    d = D * -slope

    # calculate the steering amount
    steer = int(round(p+i+d))
    steer = clamp(steer, -10, 10)

    cmd_q.put(ControlCommand('steer', steer))

    #if abs(steer) > 9:
    #  cmd_q.put(ControlCommand('speed', SPEED_TURN))
    #else:
    #  cmd_q.put(ControlCommand('speed', SPEED_NORMAL))


def to_error(ll, rr, th):
  '''Convert a left_x, right_x, angle into an error term.'''
  if None in (ll, rr, th):
    return None
  return 0.5 - (rr+ll)/2

def get_recorder(out_fname, codec='XVID'):
  '''Get a cv2 VideoWriter instance.'''
  major_ver, minor_ver, subminor_ver = cv2.__version__.split('.')[:3]

  if int(major_ver) < 3 :
    fourcc = cv2.cv.CV_FOURCC(*codec)
  else:
    fourcc = cv2.VideoWriter_fourcc(*codec)

  return cv2.VideoWriter(out_fname, fourcc, 20.0, (640, 480))

