import sys
sys.path.append(".")
from pattern import Checker, Circle, Spectrum
import generator

from pattern import Checker, Circle, Spectrum

checker = Checker(600,30)
checker.draw()
checker.show()

c = Circle(1024, 200, (512, 256))
c.draw()
c.show()

spectrum = Spectrum(256)
spectrum.draw()
spectrum.show()

gen = generator.ImageGenerator('./exercise_data/', './Labels.json', 30, [32, 32, 3], rotation=False,mirroring=False, shuffle=False)
gen.next()
gen.show()
