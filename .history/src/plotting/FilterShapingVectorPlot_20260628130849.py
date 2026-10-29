import numpy as np; import matplotlib.pyplot as plt

from src.filter_designer.filter_helper import FIRShaper

delay = 44100*0.01
I = 2**14
L = I+delay

x_0 = FIRShaper(L,I,0)
x_1 = F