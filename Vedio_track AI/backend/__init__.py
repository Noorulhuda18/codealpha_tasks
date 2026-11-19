"""Computer-vision backend for VisionTrack AI.

Nothing in this package imports Streamlit. Heavy dependencies (Ultralytics,
PyTorch) are imported lazily inside the functions that need them, so the
package can be imported — and most of it tested — without them.
"""
