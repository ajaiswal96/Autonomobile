class CursesMock(object):
  def __init__(self):
    self.cur = []
  def addstr(self, _, __, s):
    self.cur.append(s)
  def refresh(self):
    print ''.join(self.cur)
    print
    self.cur = []
