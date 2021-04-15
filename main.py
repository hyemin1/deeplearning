import sys

sys.path.append(".")

from pattern import Checker, Circle

checker = Checker(600,30)
checker.draw()
checker.show()

c = Circle(1024, 200, (512, 256))
c.draw()
c.show()
