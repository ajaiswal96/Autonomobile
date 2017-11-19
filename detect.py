import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import gauss


DEBUG = True

# How far down the horizon is
HORIZON = 0.4

# Matrix to transform the image into a topdown view
TOPDOWN = (
   0.33, 0.67,   # top left, top right
   0.00, 1.00,   # bottom left, bottom right
)

# nxn - how big the kernels are
KERN_SIZE = 10

# nxn - how big the bilateral kernels are
BLUR_KERN_SIZE = 4

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

  if DEBUG:
    # show a debug frame
    fr_dbg = fr.copy()
    cv2.line(fr_dbg, tuple(src[0]), tuple(src[3]), (0, 0, 255), 4)
    cv2.line(fr_dbg, tuple(src[1]), tuple(src[2]), (0, 0, 255), 4)
    cv2.imshow('to transform', fr_dbg)

  mat, status = cv2.findHomography(src, dst)
  trans = cv2.warpPerspective(fr, mat, (w, h))
  result = cv2.resize(trans, (w, h*1))

  return result

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
  return cv2.Canny(fr, 100, 1100)

def lanes(fr):

  ########## Separable x-y ##########

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
  sf = max(
    abs(fra.min()), abs(fra.max()),
    abs(frb.min()), abs(frb.max()),
    abs(frc.min()), abs(frc.max()),
    abs(frd.min()), abs(frd.max()),
  )

  fra /= sf
  frb /= sf
  frc /= sf
  frd /= sf

  ########## Actual 2D ##########

  # get the 2nd deriv gaussian basis kernels
  g2ha = gauss.gauss2d(KERN_SIZE, gtype='2h', direction='a')
  g2hb = gauss.gauss2d(KERN_SIZE, gtype='2h', direction='b')
  g2hc = gauss.gauss2d(KERN_SIZE, gtype='2h', direction='c')
  g2hd = gauss.gauss2d(KERN_SIZE, gtype='2h', direction='d')

  # compute the basis responses
  fra_kern = cv2.filter2D(fr, cv2.CV_32FC1, g2ha)
  frb_kern = cv2.filter2D(fr, cv2.CV_32FC1, g2hb)
  frc_kern = cv2.filter2D(fr, cv2.CV_32FC1, g2hc)
  frd_kern = cv2.filter2D(fr, cv2.CV_32FC1, g2hd)

  # scale the basis responses to a reasonable range
  sf = max(
    abs(fra_kern.min()), abs(fra_kern.max()),
    abs(frb_kern.min()), abs(frb_kern.max()),
    abs(frc_kern.min()), abs(frc_kern.max()),
    abs(frd_kern.min()), abs(frd_kern.max()),
  )

  fra_kern /= sf
  frb_kern /= sf
  frc_kern /= sf
  frd_kern /= sf

  for th in np.linspace(0, 4*np.pi, num=4*180):
    print '============='
    result = (
      + 1.0 * np.cos(th)**3              * fra
      - 3.0 * np.cos(th)**2 * np.sin(th) * frb
      + 3.0 * np.sin(th)**2 * np.cos(th) * frc
      - 1.0 * np.sin(th)**3              * frb
    )

    print 'ORIG %0.2f %0.2f' % (result.min(), result.max())

    result = np.absolute(result)
    result = np.clip(result, 0, 1)
    result = (result*255).astype(np.uint8)

    # draw a line
    x0, y0 = 100, 100
    x1, y1 = int(x0 + 100 * np.cos(th)), int(y0 - 100 * np.sin(th))
    cv2.line(result, (x0, y0), (x1, y1), (255, 255, 255), 4)

    cv2.imshow('steer', result)

    print '%+02d' % (th / np.pi * 180)

    if chr(cv2.waitKey() & 0xff) == 'q':
      break

  return result

def hough(fr, orig):
  lines = cv2.HoughLines(fr,1,np.pi/180,200)
  for line in lines[:5]:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(fr,(x1,y1),(x2,y2),(0,0,255),2)
  return fr

def detect_lanes(fr):
  pipeline = (
    crop_road,
    blur,
    to_grayscale,
    lanes,
    #transform_topdown,
    #edges,
    #hough,
  )

  if DEBUG:
    print '================='
    cv2.imshow('original', fr)

  total_time = 0

  for img_step in pipeline:
    start = time.time()
    fr = img_step(fr)
    end = time.time()
    stage_time = end-start
    total_time += stage_time
    if DEBUG:
      print '-->', int(stage_time*1000), img_step.__name__
      cv2.imshow(img_step.__name__, fr)

  if DEBUG:
    print 'FPS', round(1.0 / total_time, 1)

def main():
  #vid = cv2.VideoCapture('driving_long.mp4')
  #vid.set(1, 40000)
  #vid = cv2.VideoCapture('test_track_2.mkv')
  #vid.set(1, 230)
  vid = cv2.VideoCapture('test_track_3.mkv')
  #vid.set(1, 200)
  #vid = cv2.VideoCapture('para.jpg')

  while True:
    ret, fr = vid.read()
    if not ret:
      break

    detect_lanes(fr)

    if chr(cv2.waitKey() & 0xff) == 'q':
      break

if __name__ == '__main__':
  main()
