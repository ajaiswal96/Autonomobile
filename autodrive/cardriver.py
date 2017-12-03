import termios
import sys, tty

PWM_DUTY_CYCLE = '/sys/class/pwm/pwmchip0/pwm%s/duty_cycle'

PWM_STEERING = '0'
PWM_SPEED = '2'

PERIOD = 10000000
LOWER  =  1000000
UPPER  =  2000000
STEP   =    40000

SPEED_INC = 4
DIRECTION_INC = -4

cur_speed = 0
cur_dir = 0

def write_cmd(gpio_num, value):
  assert gpio_num == PWM_STEERING or gpio_num == PWM_SPEED
  assert -10 <= value <= 10
  duty_cycle = (UPPER+LOWER)/2 + value*STEP
  #if gpio_num == PWM_STEERING: print 'steer:', duty_cycle
  #else: print 'speed:', duty_cycle
  with open(PWM_DUTY_CYCLE % gpio_num, 'wt') as f:
    f.write(str(duty_cycle))

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

# -10 -> left, +10 -> right, 0 -> center
def set_steering(val):
  cur_dir = val
  write_cmd(PWM_STEERING, -val)
  set_speed(cur_speed)

# -10 -> reverse, +10 -> forward, 0 -> stop
def set_speed(val):
  global cur_speed
  cur_speed = val
  if abs(cur_dir) > 7: val += 1
  write_cmd(PWM_SPEED, val)

# make the car stop everything
def reset():
  write_cmd(PWM_STEERING, 0)
  write_cmd(PWM_SPEED, 0)

def main():
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

  set_speed(0)
  set_steering(0)
      
if __name__ == '__main__':
  main()
