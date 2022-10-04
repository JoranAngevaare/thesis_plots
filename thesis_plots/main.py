import matplotlib.pyplot as plt

def setup_plt():
    # Change the plots to have style and fashion
    params = {'axes.grid': False,
              #           'font.size': 16,
              #           'axes.titlesize': 22,
              #           'axes.labelsize': 18,
              #           'axes.linewidth': 2,
              #           'xtick.labelsize': 16,
              #           'ytick.labelsize': 16,
              'font.size': 16,
              'axes.titlesize': 18,
              'axes.labelsize': 16,
              'axes.linewidth': 2,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'ytick.major.size': 8,
              'ytick.minor.size': 4,
              'xtick.major.size': 8,
              'xtick.minor.size': 4,
              'xtick.major.width': 2,
              'xtick.minor.width': 2,
              'ytick.major.width': 2,
              'ytick.minor.width': 2,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'legend.fontsize': 16,
              'figure.facecolor': 'w',
              'figure.figsize': (12, 8),
              'image.cmap': 'viridis',
              }
    plt.rcParams.update(params)

