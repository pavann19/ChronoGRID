"""Read trusted local research artifacts with their Windows loss-module reference."""
import importlib
import sys

import joblib


def load_research_model(path):
    # Windows-built sklearn artifacts reference the Cython extension as _loss;
    # Linux exposes the same implementation under sklearn._loss._loss.
    # These are trusted repository artifacts, never uploaded/untrusted pickles.
    sys.modules.setdefault('_loss', importlib.import_module('sklearn._loss._loss'))
    return joblib.load(path)
