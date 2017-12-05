'''
Utilities for worker controllers.
'''

class ControlCommand(object):
  '''A control command.
 
  cmdtype must be one of:
    start
    stop
    steer
    speed
  '''
  def __init__(self, cmdtype, value=0):
    assert cmdtype in ('start', 'stop', 'steer', 'speed')
    assert -10 <= value <= 10
    self.cmdtype = cmdtype
    self.value = value


def get_img(img_req, img_q):
  '''Get an image from the master camera process.'''
  img_req.send(True)
  return img_q.get()


def clamp(val, lower, upper):
  if val < lower: return lower
  if val > upper: return upper
  return val
