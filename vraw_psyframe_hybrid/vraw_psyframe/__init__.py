# vraw_psyframe_hybrid/vraw_psyframe/__init__.py

from .vraw_encoder import VRAWEncoder
from .chids import CHIDS
from .adaptive_controller import AdaptiveController
from .adaptive_aes import AdaptiveAES
from .hybrid_processor import HybridFrameProcessor
from .utils import compress_frame_jpeg, decompress_frame_jpeg

__version__ = "0.1.0"

__all__ = [
    "VRAWEncoder",
    "CHIDS",
    "AdaptiveController",
    "AdaptiveAES",
    "HybridFrameProcessor",
    "compress_frame_jpeg",
    "decompress_frame_jpeg"
]

print(f"VRA Ψ-Frame Hybrid System Initialized (Version: {__version__})")
