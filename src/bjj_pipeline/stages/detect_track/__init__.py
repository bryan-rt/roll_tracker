"""Role: detection + tracking stage package."""

from .detector import *
from .tracker import *
from .quality import *
from .outputs import *

# IMPORTANT:
# Do NOT import processor at module import time; orchestration tests import this
# package just to reach run.py. Processor is imported by run.py when needed.
