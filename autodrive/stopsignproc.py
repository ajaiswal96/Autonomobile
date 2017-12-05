'''
Stop sign detector worker
'''

from sklearn.cluster import MiniBatchKMeans
from workerutil import ControlCommand
from workerutil import get_img

import cv2
import numpy as np
import time

STOPSIGN_PARAMS = 'resources/stopsign_classifier.xml'

SIGN_SIZE_THRESHOLD = 0.1

def stopsign_proc(img_req, img_q, cmd_q):
  '''The stop sign worker process.'''
  classifier = cv2.CascadeClassifier(STOPSIGN_PARAMS)

  while True:
    # get a frame, and save a copy of it for later
    fr = get_img(img_req, img_q)
    fr = cv2.resize(fr, (320, 240))
    w, h = fr.shape[:2]

    color_fr = fr.copy()

    # do the cascade classifier
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    signs = classifier.detectMultiScale(fr, 1.02, 10)

    # classify the signs
    largest_sign = -1
    for x, y, w, h in signs:
      subfr = color_fr[y:y+h, x:x+w, :]
      subft = cv2.resize(subfr, (25, 25))
      if is_stopsign(subfr) and w*h > largest_sign:
        largest_sign = w*h

    # saw a big stop sign, so stop for a bit
    if float(largest_sign) / w > SIGN_SIZE_THRESHOLD:
      cmd_q.put(ControlCommand('start'))
      cmd_q.put(ControlCommand('speed', 0))
      time.sleep(2)
      cmd_q.put(ControlCommand('stop'))
      time.sleep(5)

def is_stopsign(fr):
  '''Determine if a frame contains a stop sign'''
  quant, labels, centers = quantize(fr)
  hsv = [lab2hsv(ctr) for ctr in centers]
  for h, s, v in (lab2hsv(c) for c in centers):
    if h < 10 or 140 < h <= 180 and s > 50 and v > 50:
      return True
  return False

def quantize(fr, num_colors=2):
  '''Quantize an image into two colors.'''
  h, w = fr.shape[:2]

  # LAB colorspace works better for quantization
  fr = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)

  # do the k-means
  clt = MiniBatchKMeans(n_clusters = num_colors)
  labels = clt.fit_predict(fr.reshape((h*w, 3)))
  centers = clt.cluster_centers_.astype(np.uint8)

  # construct the quantized image, and go back to BGR
  quant = centers[labels].reshape((h, w, 3))
  quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

  return quant, labels, centers

def lab2hsv(pix):
  '''Convert a pixel value from the LAB colorspace to HSV'''
  fr = np.array([[pix]], dtype=np.uint8)
  fr = cv2.cvtColor(fr, cv2.COLOR_LAB2BGR)
  fr = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
  return fr[0][0]
