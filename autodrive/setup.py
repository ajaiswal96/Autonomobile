PWM_STEERING = '2'
PWM_SPEED = '0'

GPIO_STEERING = '165'
GPIO_SPEED = '163'

PERIOD = 10000000
LOWER = 1000000
UPPER = 2000000

def setup_pwm():
  for gpio, pwm in [(GPIO_STEERING, PWM_STEERING),
                    (GPIO_SPEED, PWM_SPEED)]:
    # configure gpio to be output
    with open('/sys/class/gpio/export', 'wt') as f:
      f.write(gpio)
    with open('/sys/class/gpio/gpio%s/direction' % gpio, 'wt') as f:
      f.write('out')
    with open('/sys/class/gpio/unexport', 'wt') as f:
      f.write(gpio)

    # enable pwm and set the initial values
    with open('/sys/class/pwm/pwmchip0/export', 'wt') as f:
      f.write(pwm)
    with open('/sys/class/pwm/pwmchip0/pwm%s/period' % pwm, 'wt') as f:
      f.write(str(PERIOD))
    with open('/sys/class/pwm/pwmchip0/pwm%s/duty_cycle' % pwm, 'wt') as f:
      f.write(str((LOWER + UPPER) / 2))
    with open('/sys/class/pwm/pwmchip0/pwm%s/enable' % pwm, 'wt') as f:
      f.write(str(1))

if __name__ == '__main__':
  setup_pwm()
