import matplotlib.pyplot as plt
import numpy as np

KERN_CACHE = dict()

# Gaussian
def g(x, y): return np.exp(-(x**2+y**2))

########## First derivative Gaussian ##########
def g1a(x, y): return -2.0 * x * g(x, y)  # wrt x
def g1b(x, y): return -2.0 * y * g(x, y)  # wrt y

########## Second derivative Gaussian ##########
def g2a(x, y): return 0.9213 * (2.0*x**2-1.0) * g(x, y)
def g2b(x, y): return 1.8430 *        (x * y) * g(x, y)
def g2c(x, y): return 0.9213 * (2.0*y**2-1.0) * g(x, y)
def g21(tap): return g2a(0, tap)
def g22(tap): return g(tap, 0)
def g23(tap): return g2b(1, tap) * 2

########## Second derivative Gaussian, Hilbert transformed ##########
def g2ha(x, y): return 0.978 * (-2.254*x + x**3)     * g(x, y)
def g2hb(x, y): return 0.978 * (-0.7515  + x**2) * y * g(x, y)
def g2hc(x, y): return 0.978 * (-0.7515  + y**2) * x * g(x, y)
def g2hd(x, y): return 0.978 * (-2.254*y + y**3)     * g(x, y)
def g2h1(tap): return g2ha(tap, 0)
def g2h2(tap): return g(tap, 0)
def g2h3(tap): return g2hb(0, -tap) * np.e/2.0
def g2h4(tap): return g2hb(tap, 1) * np.e

# Create a size x size 2D Gaussian array
def gauss2d(size, gtype, direction):
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
