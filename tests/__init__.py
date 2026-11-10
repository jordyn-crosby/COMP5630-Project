# Puts src/ on the import path so the tests can import main, dataPrep and plot_results directly.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
