import matplotlib.pyplot as plt

SAVE_FORMATS = ['png', 'svg', 'pdf']

def pltsaveall(name):
    """
    A small wrapper around ``matplotlib.pyplot.savefig()`` that saves
    the plot in multiple formats with tight bounding box.
    """
    for ext in SAVE_FORMATS:
        plt.savefig(f'../images/{name}.{ext}', bbox_inches='tight')
