"""
"""
# Set basic matplotlib settings
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 13

# Import from custom scripts
from .load_data import *
from .utils import *
from .discretize import *
from .analysis import *

# We only want to import the sleap postprocessing scripts if we have
# sleap installed
try:
    from .sleap_postprocess import *

except:
    print('From tracker/_init__.py:')
    print('SLEAP not detected; make sure to install SLEAP if intending to use postprocessing notebook. Otherwise, ignore this message.')


__version__ = '0.1.0'
