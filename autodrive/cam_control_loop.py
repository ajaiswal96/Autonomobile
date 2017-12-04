'''
Main camera and drive control loop

Reads images from the camera, sends them to child processes running on separate
cores, and accepts commands from the children.
'''

from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import Queue
from threading import Thread
from threading import Event
from Queue import Empty
from workerutil import ControlCommand

import cv2
import time
import signal


# OpenCV camera ID
CAM_ID = 1

# Video capture frames per second
FPS = 60

# Controlling subprocesses
SUBPROCESSES = (
)


class ImageProcessor(object):
  '''A worker process that accepts camera frames, and emits driving commands.'''

  def __init__(self, target):
    self.target = target
    self.req_sock, self.child_req_sock = Pipe()
    self.img_q = Queue(1)
    self.ctrl_q = Queue()
    self.proc = None

  def start(self):
    self.proc = Process(
      target=self.target,
      args=(self.child_req_sock, self.img_q, self.ctrl_q),
      name=self.target.__name__
    )
    self.proc.start()

  def join(self):
    self.proc.join()

def capture_loop(camera, subprocesses, run):
  '''Reads frames from the camera, and forwards them to the workers.'''

  while run.is_set():
    # grab a frame
    ret, fr = camera.read()

    if not ret:
      raise IOError('Failed to read camera')

    # send the frame to all children
    for sp in subprocesses:
      if sp.req_sock.poll():
        sp.img_q.put(fr)
        sp.req_sock.recv()


def control_loop(subprocesses, run, freq=10):
  '''Reads commands from workers, and applies them to the motors.'''

  class WorkerState(object):
    '''The current set of a worker process' commands'''

    def __init__(self):
      self.active = False
      self.steer = 0
      self.speed = 0

    def __repr__(self):
      state = '  ACTIVE' if self.active else 'INACTIVE'
      return '%s %+03d %+03d' % (state, self.speed, self.steer)

  # the current state of all of the workers
  ws = [WorkerState() for _ in subprocesses]

  while run.is_set():
    start_time = time.time()

    for i, sp in enumerate(subprocesses):
      while True:
        # get the command
        try: cmd = sp.ctrl_q.get_nowait()
        except Empty: break

        # interpret the command
        assert isinstance(cmd, ControlCommand)
        if   cmd.cmdtype == 'start': ws[i].active = True
        elif cmd.cmdtype == 'stop' : ws[i].active = False
        elif cmd.cmdtype == 'steer': wd[i].steer = cmd.value
        elif cmd.cmdtype == 'speed': wd[i].speed = cmd.value
        else: raise TypeError('Invalid ControlCommand type')

    print ws

    end_time = time.time()
    wait_time = 1.0 / freq - (end_time - start_time)

    if wait_time > 0: time.sleep(wait_time)


def main():
  '''Initialize the camera, spin up the workers, and start the main loop'''

  # open the camera
  cam = cv2.VideoCapture(CAM_ID)
  cam.set(cv2.CAP_PROP_FPS, FPS)

  # the worker processes
  sps = [ImageProcessor(sp) for sp in SUBPROCESSES]

  # signal for the cam/control threads to stop
  run = Event()
  run.set()

  # this thread handles the camera
  cap = Thread(target=capture_loop, args=(cam, sps, run), name='capture')

  # this thread handles the controls
  con = Thread(target=control_loop, args=(sps, run), name='control')

  # spin up the threads and workers
  cap.start()
  con.start()

  for sp in sps:
    sp.start()

  # wait for an interrupt, and shut everything down
  try:
    signal.pause()
  except:
    run.clear()
    cap.join()
    con.join()
    for sp in sps: sp.join()

if __name__ == '__main__':
  main()
