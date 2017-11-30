import detect
import cardriver
import cv2
from PID import PID
import numpy as np
from collections import deque
import time

CAM_ID = 0

P, I, D = 0.2, 0.0, 0.02

def to_error(ll, rr, th):
  if abs(th) > 35:
    return -th/90.0
  return 0.5 - (rr+ll)/2
  #th /= 20
  #return th

def clamp(val, lower, upper):
  if val < lower: return lower
  if val > upper: return upper
  return val

def get_err(cam):
  ret, fr = cam.read()
  if not ret: return None
  ll, rr, th = detect.detect_lanes(fr)
  if ll is None or rr is None or th is None or np.isnan(ll): return None
  return to_error(ll, rr, th)

def main():
  print 'loading camera...'
  cam = cv2.VideoCapture(CAM_ID)
  ret, _ = cam.read()
  assert ret
  print 'camera loaded'

  cardriver.reset()

  raw_input()

  cardriver.set_speed(4)

  try:
    while True:
      err = get_err(cam)
      if err is None: continue
      steer = int(round(err*80))
      steer = clamp(steer, -10, 10)
      print err, steer
      cardriver.set_steering(-steer)
  finally:
    cardriver.reset()

if __name__ == '__main__':
  main()
