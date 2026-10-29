import numpy as np; import matplotlib.pyplot as plt

from src.filter_designer.filter_helper import FIRShaper

Fs = 44100
delay = int(Fs*0.01)
I = 2**14
L = I+delay

plot_time = 0.1
plot_time_samples = plot_time * Fs

x_1 = FIRShaper(L,I,0, '')
x_2 = FIRShaper(L,I,delay, '')

x1_cut = x_1[:plot_time_samples]
x2_cut = x_1[:plot_time_samples]


fig, (axes1) = plt.subplots(1, 1,figsize=(9,3))

t = [x / Fs for x in range(len(x_1))]

axes1.plot(t, x_1, color = 'black')
axes1.plot(t, x_2, color = 'grey')

axes1.set_ylabel("Weight", fontsize = 16)
axes1.set_xlabel("Time (s)", fontsize = 16)

axes1.set_yscale('log')
plt.tight_layout()
# plt.show()


plt.savefig('/Users/jamesbrookspark/Documents/03_PhD/02_Journals/03_RoomAugmentation/01_RoomAugmentation_Latex/Figures/weighting_plot.pdf')
# plt.show()