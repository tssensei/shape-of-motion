"""Shared modal-spectrum comparison controller.

The controller remains independent of Gradio at import time; the Gradio package is
loaded lazily only by the application-level ``make_demo`` function.
"""

from modal_peak_pick.apps.compare_reconstruction_ui import (
    SpectrumComparisonController,
)


__all__ = ["SpectrumComparisonController"]
