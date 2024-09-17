"""MotleyCrew root package."""
from importlib import metadata

from .crew import MotleyCrew
from .tasks import Task

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
