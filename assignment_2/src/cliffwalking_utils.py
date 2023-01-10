import matplotlib.pyplot as plt
import numpy as np

def plot_results(range1, range2, range1_label, range2_label, train_results, test_results):
  fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, sharex='all', sharey='all',
                           figsize=(6.97, 6.97 / 1.618))
  X, Y = np.meshgrid(range1, range2)
  for ax, Z, title in zip(axes, [train_results, test_results], ['training phase', 'test phase']):
    ax.plot_surface(X, Y, Z.mean(-1), cmap='coolwarm')
    ax.set_title(title)
    ax.set_xlabel(range1_label)
    ax.set_ylabel(range2_label)
    ax.set_zlabel('reward')

  plt.suptitle('Deterministic SARSA')
  plt.show()
