import detect
import cardriver
import cv2
import numpy as np
from collections import deque
import time

CAM_ID = 0

SAMPLE_LEN = 3

P, I, D = 40, 0, 10

def to_error(ll, rr, th):
  if abs(th) > 35:
    return -th/90.0
  return 0.5 - (rr+ll)/2

def clamp(val, lower, upper):
  if val < lower: return lower
  if val > upper: return upper
  return val

def get_err(cam, rec):
  ret, fr = cam.read()
  if not ret: return None
  ll, rr, th = detect.detect_lanes(fr, recorder=rec)
  if ll is None or rr is None or th is None or np.isnan(ll): return None
  return to_error(ll, rr, th)

def main():
  print 'loading camera...'
  cam = cv2.VideoCapture(CAM_ID)
  ret, _ = cam.read()
  assert ret
  print 'camera loaded'

  fourcc = cv2.cv.CV_FOURCC(*'XVID')
  recorder = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

  cardriver.reset()

  raw_input()

  cardriver.set_speed(3)

  err_hist = deque()
  time_hist = deque()

  try:
    while True:
      err = get_err(cam, recorder)
      now = time.time()
      if err is None: continue

      err_hist.append(err)
      time_hist.append(now)

      #while now - time_hist[0] > TIME_HIST:
      while len(time_hist) > SAMPLE_LEN:
        time_hist.popleft()
        err_hist.popleft()

      slope = np.polyfit(time_hist, err_hist, 1)[0]
      total = np.sum(err_hist)

      p = P * -err_hist[-1]
      i = I * -total / len(err_hist)
      d = D * -slope

      print '%0.1f %0.1f %0.1f' % (p, i, d)

      steer = int(round(p+i+d))

      steer = clamp(steer, -10, 10)

      #print '%0.2f %0.2f %d' % (err, slope, steer)
      cardriver.set_steering(steer)
  finally:
    cardriver.reset()

if __name__ == '__main__':
  main()
