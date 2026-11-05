"""Setuptools hooks for assembling the workbench's reviewed runtime payload."""
from pathlib import Path
import sys

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workbench_build import ApplicationBuild, ApplicationSdist

setup(cmdclass={'build_py': ApplicationBuild, 'sdist': ApplicationSdist})
