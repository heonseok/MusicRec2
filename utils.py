import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Drawer():
    def __init__(self):
        self.fig = plt.figure(figsize=(4, 4))

    def plot(self, samples):
        self.fig.clf()
        gs = gridspec.GridSpec(4,4)
        gs.update(wspace=0.05, hspace=0.05)

        for i,sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        """
        f,a = plt.subplots(1, 16, figsize=(16,1))
        for i in range(16):
            a[i].imshow(np.reshape(samples[i], (28,28)))
        #f.show()
        """
        return self.fig



