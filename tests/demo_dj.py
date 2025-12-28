import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from qsiml import DeutschJozsa

def run_dj():
    print("Running Deutsch-Jozsa...")
    dj = DeutschJozsa(n=3)
    dj.deutsch_jozsa()

if __name__ == "__main__":
    run_dj()
