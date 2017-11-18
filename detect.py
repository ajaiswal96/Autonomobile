import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time


DEBUG = True

# How far down the horizon is
HORIZON = 0.4

# Matrix to transform the image into a topdown view
TOPDOWN = (
   0.33, 0.67,   # top left, top right
   0.00, 1.00,   # bottom left, bottom right
)

# nxn - how big the kernels are
KERN_SIZE = 15

KERN_CACHE = dict()

# nxn - how big the bilateral kernels are
BLUR_KERN_SIZE = 4

def gauss2d(size, gtype, direction):
  def g(x, y):
    return np.exp(-(x**2+y**2))

  def g1a(x, y):
    return -2.0 * x * g(x, y)
  def g1b(x, y):
    return -2.0 * y * g(x, y)

  def g2a(x, y):
    return 0.9213 * (2.0*x**2-1.0) * g(x, y)
  def g2b(x, y):
    return 1.843 * x * y * g(x, y)
  def g2c(x, y):
    return 0.9213 * (2.0*y**2-1.0) * g(x, y)

  def g2ha(x, y):
    return 0.978 * (-2.254*x + x**3) * g(x, y)
  def g2hb(x, y):
    return 0.978 * (-0.7515 + x**2) * y * g(x, y)
  def g2hc(x, y):
    return 0.978 * (-0.7515 + y**2) * x * g(x, y)
  def g2hd(x, y):
    return 0.978 * (-2.254*y + y**3) * g(x, y)

  key = gtype + ':' + direction + str(size)
  if key in KERN_CACHE:
    return KERN_CACHE[key]

  gfn = {
    '1:a': g1a,
    '1:b': g1b,

    '2:a': g2a,
    '2:b': g2b,
    '2:c': g2c,

    '2h:a': g2ha,
    '2h:b': g2hb,
    '2h:c': g2hc,
    '2h:d': g2hd,
  }[gtype + ':' + direction]

  xx = np.linspace(-2.3, +2.3, num=size)
  yy = np.linspace(+2.3, -2.3, num=size)

  result = np.zeros((size, size), dtype=np.float32)

  for i in xrange(size):
    for j in xrange(size):
      x = xx[j]
      y = yy[i]
      result[j][i] = gfn(x, y)

  result = result / np.absolute(result).sum() * 8

  KERN_CACHE[key] = result

  return result

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

def gauss(fr):
  # get the 2nd deriv gaussian basis kernels
  g2ha = gauss2d(KERN_SIZE, gtype='2h', direction='a')
  g2hb = gauss2d(KERN_SIZE, gtype='2h', direction='b')
  g2hc = gauss2d(KERN_SIZE, gtype='2h', direction='c')
  g2hd = gauss2d(KERN_SIZE, gtype='2h', direction='d')

  # compute the basis responses
  fra = cv2.filter2D(fr, cv2.CV_32FC1, g2ha)
  frb = cv2.filter2D(fr, cv2.CV_32FC1, g2hb)
  frc = cv2.filter2D(fr, cv2.CV_32FC1, g2hc)
  frd = cv2.filter2D(fr, cv2.CV_32FC1, g2hd)

  # scale the basis responses to a reasonable range
  sf = max(
    abs(fra.min()), abs(fra.max()),
    abs(frb.min()), abs(frb.max()),
    abs(frc.min()), abs(frc.max()),
    abs(frd.min()), abs(frd.max()),
  ) / 2

  fra /= sf
  frb /= sf
  frc /= sf
  frd /= sf

  for th in np.linspace(0, 4*np.pi, num=4*180):
    print '============='

    scale = (
      + abs(1.0 * np.cos(th)**3             )
      + abs(3.0 * np.cos(th)**2 * np.sin(th))
      + abs(3.0 * np.sin(th)**2 * np.cos(th))
      + abs(1.0 * np.sin(th)**3             )
    )

    result = np.clip(
      + 1.0 * np.cos(th)**3              * fra
      - 3.0 * np.cos(th)**2 * np.sin(th) * frb
      + 3.0 * np.sin(th)**2 * np.cos(th) * frc
      - 1.0 * np.sin(th)**3              * frb
    , -1.0, 1.0)

    print 'ORIG %0.2f %0.2f' % (result.min(), result.max())

    result = (np.absolute(result) * 255).astype(np.uint8)

    #result = cv2.Canny(result, 100, 1100)

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
    gauss,
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

  while True:
    ret, fr = vid.read()
    if not ret:
      break

    detect_lanes(fr)

    if chr(cv2.waitKey() & 0xff) == 'q':
      break

if __name__ == '__main__':
  main()
