"""Evaluate image tood on the test split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.tood.build import build_model

if __name__ == "__main__":
    main("evaluate", task="image", version="tood", build_model=build_model)
