import termios
import sys, tty
import time
from threading import Thread
from threading import Event

PWM_DUTY_CYCLE = '/sys/class/pwm/pwmchip0/pwm%s/duty_cycle'

PWM_STEERING = '0'
PWM_SPEED = '2'

PERIOD = 10000000
LOWER  =  1000000
UPPER  =  2000000
STEP   =    40000

cur_speed = 0
speedset_run = Event()
speedset_thr = None

def speedset():
  '''Background thread that polls the current speed, and does PWM on the PWM to
  accomadate float values.'''
  period = 0.05
  while not speedset_run.is_set():
    start = time.time()

    _cur_speed = cur_speed

    # round almost-int cases
    if abs(_cur_speed - round(_cur_speed)) < 0.01:
      _cur_speed = int(round(_cur_speed))

    # handle the integer cases
    if isinstance(_cur_speed, int):
      write_cmd(PWM_SPEED, _cur_speed)

    # pwm the pwm
    else:
      direction = -1 if _cur_speed < 0 else +1
      if direction == -1: _cur_speed *= -1.0

      # alternate between floor(spd) and floor(spd)+1
      base_speed = int(_cur_speed)
      fast_speed = base_speed + 1

      fast_time = (_cur_speed - base_speed) * period
      base_time = period - fast_time

      # put on floor(spd) for a little bit...
      t0 = time.time()
      write_cmd(PWM_SPEED, base_speed * direction)
      t1 = time.time()
      sleep_time = base_time - (t1-t0)
      if sleep_time > 0: time.sleep(sleep_time)

      # ... then put on floor(spd)+1 for the rest of the period
      t2 = time.time()
      write_cmd(PWM_SPEED, fast_speed * direction)
      t3 = time.time()
      sleep_time = base_time - (t3-t2)
      if sleep_time > 0: time.sleep(sleep_time)

    end = time.time()

    # wait until the period is over
    wait_duration = period - (end-start)
    if wait_duration > 0:
      time.sleep(wait_duration)

  # exited, so stop
  write_cmd(PWM_SPEED, 0)

def write_cmd(gpio_num, value):
  assert gpio_num == PWM_STEERING or gpio_num == PWM_SPEED
  assert -10 <= value <= 10
  duty_cycle = (UPPER+LOWER)/2 + value*STEP
  #if gpio_num == PWM_STEERING: print 'steer:', duty_cycle
  #else: print 'speed:', duty_cycle
  with open(PWM_DUTY_CYCLE % gpio_num, 'wt') as f:
    f.write(str(duty_cycle))

# -10 -> left, +10 -> right, 0 -> center
def set_steering(val):
  '''Just write the steering directly out'''
  write_cmd(PWM_STEERING, -val)

# -10 -> reverse, +10 -> forward, 0 -> stop
def set_speed(val):
  '''All we need to do here is set the current speed global var'''
  global cur_speed
  cur_speed = val

def setup():
  '''Spin up the speed-setting thread'''
  global speedset_thr
  speedset_thr = Thread(target=speedset)
  speedset_thr.start()

def stop():
  '''Terminate the speed-setting thread'''
  assert speedset_thr is not None
  speedset_run.set()
  speedset_thr.join()

# make the car stop everything
def reset():
  set_speed(0)
  set_steering(0)

def getch():
  def _getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(fd)
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
  return _getch()

def main():
  setup()
  
  speed = 0      
  direction = 0

  while True:
    cmd = getch()

    if cmd == 'w':
      speed += 1
    elif cmd == 'a':
      direction -= 1
    elif cmd == 's':
      speed -= 1
    elif cmd == 'd':
      direction += 1
    elif cmd == 'r':
      speed = 0
      direction = 0
    elif cmd == 'q':
      break

    if speed < -10: speed = -10
    elif speed > 10: speed = 10

    if direction < -10: direction = -10
    elif direction > 10: direction = 10

    set_speed(speed)
    set_steering(direction)

  stop()
  
      
if __name__ == '__main__':
  main()
