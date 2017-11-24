import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import gauss


DEBUG = False
VERBOSE = True

# How far down the horizon is
HORIZON = 0.4

# Matrix to transform the image into a topdown view
TOPDOWN = (
  +0.00, +1.00,   # top left, top right
  -1.00, +2.00,   # bottom left, bottom right
)

# nxn - how big the kernels are
KERN_SIZE = 10

# nxn - how big the bilateral kernels are
BLUR_KERN_SIZE = 4

# color of lanes
LANE_COLOR_RANGE = (
  ( 90,  70,  50),
  (130, 200, 200)
)

WIDTH = 640
HEIGHT = 480

def filter_lanecolor(fr):
  hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
  lower, upper = LANE_COLOR_RANGE
  mask = cv2.inRange(hsv, lower, upper)
  return cv2.bitwise_and(fr, fr, mask=mask)

def to_grayscale(fr):
  return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

def transform_topdown(fr):
  h, w = fr.shape

  dst = np.array([
    [0, 0], [w, 0],
    [w, h], [0, h],
  ], dtype=np.float32)

  src = np.array([
    [TOPDOWN[0]*w, 0], [TOPDOWN[1]*w, 0],
    [TOPDOWN[3]*w, h], [TOPDOWN[2]*w, h],
  ], dtype=np.float32)

  mat, status = cv2.findHomography(src, dst)
  result = cv2.warpPerspective(fr, mat, (w, h))

  return result

def untransform_topdown(fr):
  h, w = fr.shape

  dst = np.array([
    [TOPDOWN[0]*w, 0], [TOPDOWN[1]*w, 0],
    [TOPDOWN[3]*w, h], [TOPDOWN[2]*w, h],
  ], dtype=np.float32)

  src = np.array([
    [0, 0], [w, 0],
    [w, h], [0, h],
  ], dtype=np.float32)

  mat, status = cv2.findHomography(src, dst)
  return cv2.warpPerspective(fr, mat, (w, h))

def crop_road(fr):
  h, w, c = fr.shape
  new_h = int(round(h*HORIZON))
  return fr[new_h:, :]

def blur(fr):
  ITERS = 4
  for _ in xrange(ITERS):
    fr = cv2.bilateralFilter(fr, BLUR_KERN_SIZE, 30, 1000)
  return fr

def edges(fr):
  return cv2.Canny(fr, 50, 400)

def resize(fr):
  return cv2.resize(fr, (WIDTH, HEIGHT))

def lanes(fr):
  # Hilbert transform of gauss2 9-tap FIR x-y separable filters
  g2hf1 = np.array([gauss.g2h1(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32)
  g2hf2 = np.array([gauss.g2h2(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32)
  g2hf3 = np.array([gauss.g2h3(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32)
  g2hf4 = np.array([gauss.g2h4(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32)

  # compute the basis responses
  fra = cv2.sepFilter2D(fr, cv2.CV_32FC1, g2hf1, g2hf2)
  frb = cv2.sepFilter2D(fr, cv2.CV_32FC1, g2hf4, g2hf3)
  frc = cv2.sepFilter2D(fr, cv2.CV_32FC1, g2hf3, g2hf4)
  frd = cv2.sepFilter2D(fr, cv2.CV_32FC1, g2hf2, g2hf1)

  # scale the basis responses to a reasonable range
  sf = 255.0

  fra /= sf
  frb /= sf
  frc /= sf
  frd /= sf

  # mask out the transform edges
  _, w = fr.shape
  shift = KERN_SIZE
  mask = np.ones(fr.shape, dtype=np.float32)
  mask[:, 0:shift] = 0.0
  mask[:, w-shift:w] = 0.0
  mask = transform_topdown(mask)

  fra *= mask
  frb *= mask
  frc *= mask
  frd *= mask

  if DEBUG:
    cv2.imshow('fra', np.absolute(fra))
    cv2.imshow('frb', np.absolute(frb))
    cv2.imshow('frc', np.absolute(frc))
    cv2.imshow('frd', np.absolute(frd))

  # find the angle with the highest response
  max_sum = -1
  max_angle = None

  ra = abs(fra.sum())
  rb = abs(frb.sum())
  rc = abs(frc.sum())
  rd = abs(frd.sum())

  angles = np.linspace(-np.pi/2, np.pi/2, num=50)
  responses = (
    + 1.0 * np.cos(angles)**3                  * ra
    - 3.0 * np.cos(angles)**2 * np.sin(angles) * rb
    + 3.0 * np.sin(angles)**2 * np.cos(angles) * rc
    - 1.0 * np.sin(angles)**3                  * rb
  )

  max_angle = angles[np.argmax(responses)]

  # compute the final image
  result = (
    + 1.0 * np.cos(max_angle)**3                     * fra
    - 3.0 * np.cos(max_angle)**2 * np.sin(max_angle) * frb
    + 3.0 * np.sin(max_angle)**2 * np.cos(max_angle) * frc
    - 1.0 * np.sin(max_angle)**3                     * frb
  )

  result = np.absolute(result)
  result = np.clip(result, 0, 1)
  result = (result*255).astype(np.uint8)

  # draw a line
  #x0, y0 = 100, 100
  #x1, y1 = int(x0 + 100 * np.cos(max_angle)), int(y0 - 100 * np.sin(max_angle))
  #cv2.line(result, (x0, y0), (x1, y1), (255, 255, 255), 4)

  #if chr(cv2.waitKey() & 0xff) == 'b':
  #  plt.plot(xrange(50), responses)
  #  plt.show()

  return result

def hough(fr):
  lines = cv2.HoughLinesP(fr, rho=1,
    theta=np.pi/180, threshold=50,
    maxLineGap=30, minLineLength=80)
  if lines is None: lines = [[]]

  result = np.zeros(fr.shape, dtype=np.uint8)

  x0a, y0a, x1a, y1a = [], [], [], []
  for line in lines:
    if len(line) > 0:
      x0, y0, x1, y1 = line[0]
      ((y0, x0), (y1, x1)) = sorted(((y0, x0), (y1, x1)))
      x0a.append(x0)
      y0a.append(y0)
      x1a.append(x1)
      y1a.append(y1)
      cv2.line(result, (x0, y0), (x1, y1), (255, 255, 255), 1)
  
  for x, y in zip(x0a, y0a):
    cv2.circle(result, (x, y), 2, (256, 256, 256), thickness=-1)
  for x, y in zip(x1a, y1a):
    cv2.circle(result, (x, y), 2, (256, 256, 256), thickness=-1)

  if len(x0a) > 0:
    x0 = int(round(float(sum(x0a)) / len(x0a)))
    y0 = int(round(float(sum(y0a)) / len(y0a)))
    x1 = int(round(float(sum(x1a)) / len(x1a)))
    y1 = int(round(float(sum(y1a)) / len(y1a)))

    cv2.line(result, (x0, y0), (x1, y1), (255, 255, 255), 1)

  return result

def detect_lanes(fr):
  pipeline = (
    resize,
    crop_road,
    #blur,
    #filter_lanecolor,
    to_grayscale,
    transform_topdown,
    lanes,
    edges,
    hough,
    untransform_topdown,
  )

  if VERBOSE:
    print '================='
  if DEBUG:
    cv2.imshow('original', fr)

  orig = fr.copy()

  total_time = 0

  for img_step in pipeline:
    start = time.time()
    fr = img_step(fr)
    end = time.time()
    stage_time = end-start
    total_time += stage_time
    if VERBOSE:
      print '-->', int(stage_time*1000), img_step.__name__
    if DEBUG:
      cv2.imshow(img_step.__name__, fr)
    if img_step.__name__ == 'crop_road':
      inter = fr.copy()

  if DEBUG:
    cv2.imshow('final', cv2.bitwise_or(inter,np.stack([fr]*3, axis=2)))

  if VERBOSE:
    print 'FPS', round(1.0 / total_time, 1)

def main():
  #vid = cv2.VideoCapture('driving_long.mp4')
  #vid.set(1, 40000)
  #vid = cv2.VideoCapture('test_track_2.mkv')
  #vid.set(1, 230)
  vid = cv2.VideoCapture('test_track_3.mkv')
  #vid.set(1, 200)
  #vid = cv2.VideoCapture('para2.jpg')

  while True:
    ret, fr = vid.read()
    if not ret:
      break

    detect_lanes(fr)

    if DEBUG and chr(cv2.waitKey() & 0xff) == 'q':
      break

if __name__ == '__main__':
  main()
