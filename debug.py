import utils.data_loaders as data_loaders
import TimeTagger
import matplotlib.pyplot as plt
import time

file = "data/TERm1010.ttbin"
m = 2
(p, s) = data_loaders.get_correlator_architecture(alpha=7, m=m, tau_max=1e-2, t0=1e-12)
reader = TimeTagger.FileReader(file)
buffer = reader.getData(1e7)
t = buffer.getTimestamps()
ch = buffer.getChannels()
t = t[ch == 1]

start_time = time.time()
(g2, tau) = data_loaders.async_corr(t, p, m, s, tau_start=1e-7)
print("--- %s seconds ---" % (time.time() - start_time))

plt.semilogx(tau, g2)
plt.ylim([0.8, 1.7])
plt.show()
