import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
from scipy import stats
from scipy import signal
import time
import gauss
import curses
import operator
from collections import deque
from curses_mock import CursesMock

DEBUG = False
VERBOSE = False
RECORDER = None

# How far down the horizon is
HORIZON = 0.7

# Matrix to transform the image into a topdown view
TOPDOWN = (
  +0.00, +1.00,   # top left, top right
  -0.80, +1.80,   # bottom left, bottom right
)

# nxn - how big the kernels are
KERN_SIZE = 10

# nxn - how big the blur kernels are
BLUR_KERN_SIZE = 15

# color of lanes
LANE_COLOR_RANGE = (
  #( 35,  20,  50),
  #(130, 255, 255)
  #( 50,  80,  80),
  #(120, 255, 255)

  #( 35,  15,  80),
  #(120, 220, 255)

  ( 30,  40,  80),
  ( 80, 255, 255),
)

# number of samples in the steerable filter sample
STEERABLE_SAMPLES = 50

# part of frame to look for lateral centeredness
LATERAL_CUTOFF = 0.75

WIDTH = 320
HEIGHT = 240

record_frame = None

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
  cropped = fr[new_h:, :]
  global record_frame
  record_frame = cropped.copy()
  return cropped

def blur(fr):
  img = cv2.GaussianBlur(fr, (BLUR_KERN_SIZE, BLUR_KERN_SIZE), 0)
  _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return img

def edges(fr):
  return cv2.Canny(fr, 50, 500)

def resize(fr):
  return cv2.resize(fr, (WIDTH, HEIGHT))

def lanes(fr):
  if not hasattr(lanes, 'g2hf'):
    # Hilbert transform of gauss2 9-tap FIR x-y separable filters
    lanes.g2hf = (
      np.array([gauss.g2h1(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32),
      np.array([gauss.g2h2(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32),
      np.array([gauss.g2h3(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32),
      np.array([gauss.g2h4(tap) for tap in np.linspace(-2.3, 2.3, KERN_SIZE)], dtype=np.float32),
    )

  # compute the basis responses
  fra = cv2.sepFilter2D(fr, cv2.CV_32FC1, lanes.g2hf[0], lanes.g2hf[1])
  frb = cv2.sepFilter2D(fr, cv2.CV_32FC1, lanes.g2hf[3], lanes.g2hf[2])
  frc = cv2.sepFilter2D(fr, cv2.CV_32FC1, lanes.g2hf[2], lanes.g2hf[3])
  frd = cv2.sepFilter2D(fr, cv2.CV_32FC1, lanes.g2hf[1], lanes.g2hf[0])

  # scale the basis responses to a reasonable range
  sf = 255.0

  fra /= sf
  frb /= sf
  frc /= sf
  frd /= sf

  # mask out the transform edges
  if not hasattr(lanes, 'mask'):
    _, w = fr.shape
    shift = KERN_SIZE
    mask = np.ones(fr.shape, dtype=np.float32)
    mask[:, 0:shift] = 0.0
    mask[:, w-shift:w] = 0.0
    mask = transform_topdown(mask)
    lanes.mask = mask

  fra *= lanes.mask
  frb *= lanes.mask
  frc *= lanes.mask
  frd *= lanes.mask

  #if DEBUG:
  #  cv2.imshow('fra', np.absolute(fra))
  #  cv2.imshow('frb', np.absolute(frb))
  #  cv2.imshow('frc', np.absolute(frc))
  #  cv2.imshow('frd', np.absolute(frd))

  # find the angle with the highest response
  max_sum = -1
  max_angle = None

  ra = abs(fra.sum())
  rb = abs(frb.sum())
  rc = abs(frc.sum())
  rd = abs(frd.sum())

  angles = np.linspace(-np.pi/2, np.pi/2, num=STEERABLE_SAMPLES+1).astype(np.float32)[:-1]
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
  h, w = fr.shape

  # close any holes
  kern = np.ones((8,8), dtype=np.uint8)
  fr = cv2.morphologyEx(fr, cv2.MORPH_CLOSE, kern)

  # clean up the image
  #to_fit = np.argmax(fr, 1)
  #to_fit = np.zeros((1, w), dtype=np.uint8)
  #for x in xrange(w):
  #  y = np.argmax(fr, 1)
  #  to_fit[x] = y

  # fit a polynomial
  #ransac = RANSACRegressor()

  #line_x = np.arange(w)
  #line_y = ransac.predict(line_x)
  #plt.plot(line_x, line_y)
  #plt.plot(np.arange(h), to_fit)
  #plt.show()

  # get the lines
  lines = cv2.HoughLinesP(fr, rho=1,
    theta=np.pi/180, threshold=40,
    maxLineGap=h/6, minLineLength=h/6)
  if lines is None: lines = [[]]

  # organize and normalize the lines
  h, w = fr.shape
  lines_sorted = []
  for line in lines:
    for x0, y0, x1, y1 in line:
      ((y0, x0), (y1, x1)) = sorted(((y0, x0), (y1, x1)))
      lines_sorted.append((
        (float(x0)/w, float(y0)/h),
        (float(x1)/w, float(y1)/h),
      ))

  # get the angle
  ths = []
  for (x0, y0), (x1, y1) in lines_sorted:
    if y1-y0 == 0:
      if x0-x1 > 0:
        th = np.pi/2
      else:
        th = -np.pi/2
    else:
      th = np.arctan(float(x0-x1)/float(y1-y0))
    ths.append(th)
  if ths:
    th = np.degrees(np.mean(ths))
  else:
    th = None

  # find left and right by looking near bottom of frame
  thresh = h/5
  fr_lower = signal.convolve(np.mean(fr[h-thresh:h,:], axis=0), signal.hann(w/10), mode='same')
  pks = [float(pk)/w for pk in signal.find_peaks_cwt(fr_lower, np.arange(1,5))]

  now = time.time()

  if not hasattr(hough, 'prev_ll'):
    hough.prev_ll = deque()
    hough.prev_rr = deque()
    hough.prev_time = deque()
    if len(pks) == 2:
      ll, rr = sorted(pks)
    else:
      ll, rr = 0.3, 0.7
  else:
    if len(pks) >= 2:
      closest_ll = 100
      closest_rr = 100
      med_ll = np.mean(hough.prev_ll)
      med_rr = np.mean(hough.prev_rr)
      for pk in pks:
        if abs(med_ll-pk) < abs(med_ll-closest_ll): closest_ll = pk
        if abs(med_rr-pk) < abs(med_rr-closest_rr): closest_rr = pk
      if closest_rr - closest_ll < 0.25:
        if abs(med_ll-closest_ll) < abs(med_rr-closest_rr):
          ll = closest_ll
          rr = ll + np.median(hough.prev_rr)-np.median(hough.prev_ll)
        else:
          rr = closest_rr
          ll = rr - np.median(hough.prev_rr)-np.median(hough.prev_ll)
      else:
        ll = closest_ll
        rr = closest_rr
      #if abs(closest_ll-med_ll) < 0.1: ll = closest_ll
      #else: ll = med_ll
      #if abs(closest_rr-med_rr) < 0.1: rr = closest_rr
      #else: rr = med_rr
    elif len(pks) == 1:
      pk = pks[0]
      med_ll = np.median(hough.prev_ll)
      med_rr = np.median(hough.prev_rr)
      if abs(pk-med_ll) > abs(pk-med_rr):
        rr = pk
        #ll = med_ll + rr-med_rr
        ll = rr - (med_rr-med_ll)
      else:
        ll = pk
        #rr = med_rr + ll-med_ll
        rr = ll + (med_rr-med_ll)
    else:
      ll = np.median(hough.prev_ll)
      rr = np.median(hough.prev_rr)

  hough.prev_ll.append(ll)
  hough.prev_rr.append(rr)
  hough.prev_time.append(now)

  #while now - hough.prev_time[0] > 1:
  while len(hough.prev_time) > 4:
    hough.prev_ll.popleft()
    hough.prev_rr.popleft()
    hough.prev_time.popleft()

  # show the debug frame
  raw_lines = np.zeros(fr.shape, dtype=np.uint8)
  raw_lines = fr
  show_lines(raw_lines, lines_sorted, 1)
  show_points(raw_lines, [(ll, 1.0), (rr, 1.0)], 3)
  if th is not None:
    p0 = (0.5, 1.0)
    p1 = (0.5+0.5*np.sin(np.radians(th)),
          1.0-0.5*np.cos(np.radians(th)))
    show_lines(raw_lines, [(p0, p1)], 5)
  if DEBUG:
    cv2.imshow('raw_lines', raw_lines)
  if RECORDER is not None:
    out1 = cv2.resize(raw_lines, (640, 240))
    out1 = cv2.cvtColor(out1, cv2.COLOR_GRAY2BGR)
    if record_frame is not None:
      out2 = cv2.resize(record_frame, (640, 240))
    else:
      out2 = np.zeros(out1.shape)
    out = np.zeros((480, 640, 3), dtype=np.uint8)
    out[:240, :, :] = out1
    out[240:, :, :] = out2
    RECORDER.write(out)

  return ll, rr, th

def hough_old(fr):
  h, w = fr.shape

  # get the lines
  lines = cv2.HoughLinesP(fr, rho=1,
    theta=np.pi/180, threshold=40,
    maxLineGap=h/10, minLineLength=h/3)
  if lines is None: lines = [[]]

  # organize and normalize the lines
  h, w = fr.shape
  lines_sorted = []
  for line in lines:
    for x0, y0, x1, y1 in line:
      ((y0, x0), (y1, x1)) = sorted(((y0, x0), (y1, x1)))
      lines_sorted.append((
        (float(x0)/w, float(y0)/h),
        (float(x1)/w, float(y1)/h),
      ))

  raw_lines = np.zeros(fr.shape, dtype=np.uint8)

  show_lines(raw_lines, lines_sorted, 1)

  # cluster the lines and compute left and right sides
  ll, rr, th = cluster_lines_lr(lines_sorted, fr)

  if ll is None or np.isnan(ll) or np.isnan(rr):
    hough.stage_debug_text = '.'*100
  else:
    hough.stage_debug_text = (
       int(round(ll*100))*'.' + '|'
     + int(round((rr-ll)*100))*'.' + '|'
     + int(round((1.0-rr)*100))*'.'
    )

  hough.stage_debug_text += '\n'

  if th is None:
    hough.stage_debug_text += '-- degrees'
  else:
    hough.stage_debug_text += '%0.1f degrees' % th

  if ll is not None: show_points(raw_lines, zip([ll], [1.0]), 5)
  if rr is not None: show_points(raw_lines, zip([rr], [1.0]), 5)
  if th is not None:
    x0, y0 = 0.5, 1.0
    x1, y1 = x0 + np.cos(np.radians(th-90))*50, y0 + np.sin(np.radians(th-90))*50
    show_lines(raw_lines, [((x0, y0), (x1, y1))], 3)

  if DEBUG:
    cv2.imshow('hough_raw', raw_lines)

  return ll, rr, th

def invert_dict(d):
  inv = dict()
  for k, v in d.iteritems():
    if v not in inv:
      inv[v] = []
    inv[v].append(k)
  return inv

# get lane parameters
# returns (x_ll, x_rr, yaw)
# x_ll, x_rr are normalized to between 0 and 1
# yaw is an angle, where 0 is straight, positive is right, negative is left
def cluster_lines_lr(lines, fr):
  if DEBUG:
    dbg = np.zeros(fr.shape, dtype=np.uint8)

  if not lines: return None, None, None

  SAMPLE_DIST = 0.05

  # interpolate the lines
  id2line = []
  pts = []
  for i, ((x0, y0), (x1, y1)) in enumerate(lines):
    dy = y1-y0
    dx = x1-x0
    seg_len = (dx**2+dy**2)**0.5
    samples = int(round(seg_len / SAMPLE_DIST))
    pts.extend(zip(
      np.linspace(x0, x1, samples),
      np.linspace(y0, y1, samples)
    ))
    id2line.extend([i]*samples)

  # cluster them
  db = [int(i) for i in DBSCAN(eps=0.1, min_samples=5).fit(np.array(pts)).labels_]

  # map lbl -> [pt_ids]
  db_inv = invert_dict(dict(zip(xrange(len(db)), db)))

  # filter out y>LATERAL_CUTOFF
  db_inv_lower = dict()
  for lbl, pt_ids in db_inv.iteritems():
    lower = [i for i in pt_ids if pts[i][1] > LATERAL_CUTOFF]
    if lower: db_inv_lower[lbl] = lower

  # compute the average x locations
  db_x_avg = dict()
  for lbl, pt_ids in db_inv_lower.iteritems():
    x_avg = np.mean([pts[i][0] for i in pt_ids])
    db_x_avg[lbl] = x_avg

  # sort them
  xavgs = sorted(db_x_avg.iteritems(), key=operator.itemgetter(1))

  # choose the middle two
  HISTORY = 1
  now = time.time()
  if not hasattr(cluster_lines_lr, 'lls'):
    cluster_lines_lr.lls = deque()
    cluster_lines_lr.rrs = deque()
    cluster_lines_lr.times = deque()
 
  lbl_ll, ll = None, None
  lbl_rr, rr = None, None

  if len(xavgs) >= 2:
    lbl_ll, ll = xavgs[len(xavgs)/2-1]
    lbl_rr, rr = xavgs[len(xavgs)/2]
    cluster_lines_lr.lls.append(ll)
    cluster_lines_lr.rrs.append(rr)
    cluster_lines_lr.times.append(now)
    while now - cluster_lines_lr.times[0] > HISTORY:
      cluster_lines_lr.lls.popleft()
      cluster_lines_lr.rrs.popleft()
      cluster_lines_lr.times.popleft()
  elif len(xavgs) == 1:
    lbl_pp, pp = xavgs[0]
    dll = abs(pp - np.mean(cluster_lines_lr.lls))
    drr = abs(pp - np.mean(cluster_lines_lr.rrs))
    width = abs(np.mean(cluster_lines_lr.lls) - np.mean(cluster_lines_lr.rrs))
    if dll < drr:
      lbl_ll, ll = lbl_pp, pp
      lbl_rr, rr = None, pp + width
    else:
      lbl_ll, ll = None, pp - width
      lbl_rr, rr = lbl_pp, pp
  else:
    return None, None, None

  # do a linear fit on each cluster's points
  ll_pts = [pts[i] for i in (db_inv[lbl_ll] if lbl_ll is not None else [])]
  rr_pts = [pts[i] for i in (db_inv[lbl_rr] if lbl_rr is not None else [])]

  ll_linfit = None
  rr_linfit = None

  if DEBUG:
    if ll_pts: show_points(dbg, ll_pts, 1)
    if rr_pts: show_points(dbg, rr_pts, 1)

  if ll_pts: ll_linfit = np.poly1d(np.polyfit(zip(*ll_pts)[1], zip(*ll_pts)[0], 1))
  if rr_pts: rr_linfit = np.poly1d(np.polyfit(zip(*rr_pts)[1], zip(*rr_pts)[0], 1))

  if DEBUG:
    if ll_linfit: show_lines(dbg, [((ll_linfit(0.0), 0.0), (ll_linfit(1.0), 1.0))], 1)
    if rr_linfit: show_lines(dbg, [((rr_linfit(0.0), 0.0), (rr_linfit(1.0), 1.0))], 1)

  # compute the angle
  ll_theta = None
  rr_theta = None
  if ll_linfit: ll_theta = np.arctan(-ll_linfit[1]) if ll_linfit[1] != 0 else float('inf')
  if rr_linfit: rr_theta = np.arctan(-rr_linfit[1]) if rr_linfit[1] != 0 else float('inf')

  theta = None
  all_thetas = [t for t in (ll_theta, rr_theta) if t is not None]
  if all_thetas: theta = np.degrees(np.mean(all_thetas))

  if DEBUG:
    cv2.imshow('cluster-dbg', dbg)

  return ll, rr, theta

def show_points(fr, pts, th):
  h, w = fr.shape
  for x, y in pts:
    x *= w
    y *= h
    x = int(round(x))
    y = int(round(y))
    cv2.circle(fr, (x, y), th, (255, 255, 255), thickness=-1)

def show_lines(fr, lines, th):
  h, w = fr.shape
  for ((x0, y0), (x1, y1)) in lines:
    x0 *= w
    y0 *= h
    x1 *= w
    y1 *= h
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    cv2.line(fr, (x0, y0), (x1, y1), (255, 255, 255), thickness=th)

def gaussian_blur(fr):
  return cv2.GaussianBlur(fr, (25, 25), 0)

def detect_lanes(fr, screen=None, recorder=None):
  pipeline = (
    resize,
    crop_road,
    filter_lanecolor,
    to_grayscale,
    blur,
    transform_topdown,
    #lanes,
    #edges,
    hough,
    #untransform_topdown,
  )

  global RECORDER
  RECORDER = recorder

  status_str = []

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
    status_str.append('--> %d %s' % (int(stage_time*1000), img_step.__name__))
    if hasattr(img_step, 'stage_debug_text'):
      status_str.append(img_step.stage_debug_text)
    if DEBUG and img_step.__name__ != 'hough':
      cv2.imshow(img_step.__name__, fr)
    if img_step.__name__ == 'crop_road':
      inter = fr.copy()

  #if DEBUG:
  #  cv2.imshow('final', cv2.bitwise_or(inter,np.stack([fr]*3, axis=2)))

  status_str.append('FPS %.1f' % (1.0/total_time))

  if VERBOSE:
    if not hasattr(detect_lanes, 'frnum'):
      detect_lanes.frnum = 0
    screen.addstr(0, 0, 'Frame %d' % detect_lanes.frnum)
    screen.addstr(1, 0, '\n'.join(status_str)+'\n')
    screen.refresh()
    detect_lanes.frnum += 1
  
  return fr

def main(screen):
  #vid = cv2.VideoCapture('driving_long.mp4')
  #vid.set(1, 40000)
  #vid = cv2.VideoCapture('test_track_2.mkv')
  #vid.set(1, 230)
  #vid = cv2.VideoCapture('test_track_3.mkv')
  #vid.set(1, 200)
  #vid = cv2.VideoCapture('para2.jpg')
  vid = cv2.VideoCapture(0)

  while True:
    ret, fr = vid.read()
    if not ret:
      break

    detect_lanes(fr, screen)

    if DEBUG and chr(cv2.waitKey(1) & 0xff) == 'q':
      break

if __name__ == '__main__':
  VERBOSE = True
  DEBUG = True
  #screen = CursesMock()
  #main(screen)
  curses.wrapper(main)
