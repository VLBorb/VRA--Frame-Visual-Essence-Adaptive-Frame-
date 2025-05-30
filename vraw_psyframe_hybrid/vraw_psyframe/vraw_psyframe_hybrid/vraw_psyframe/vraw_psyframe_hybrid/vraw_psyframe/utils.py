# vraw_psyframe_hybrid/vraw_psyframe/utils.py

import cv2
import numpy as np

def compress_frame_jpeg(frame: np.ndarray, quality=90) -> bytes:
    """
    Compresses a frame using JPEG.
    :param frame: 2D or 3D numpy array (BGR format if color).
    :param quality: JPEG quality level (0-100).
    :return: Compressed bytes in JPEG format.
    """
    if frame is None:
        raise ValueError("Input frame cannot be None.")
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    is_success, buf = cv2.imencode(".jpg", frame, encode_params)
    if not is_success:
        raise RuntimeError("JPEG compression failed.")
    return buf.tobytes()

def decompress_frame_jpeg(comp_data: bytes) -> np.ndarray:
    """
    Decompresses JPEG bytes back to a numpy frame.
    :param comp_data: Compressed JPEG data bytes.
    :return: Decompressed frame as a numpy array (BGR format if color).
    """
    if comp_data is None:
        raise ValueError("Compressed data cannot be None.")
    npbuf = np.frombuffer(comp_data, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_ANYCOLOR) # IMREAD_ANYCOLOR preserves color
    if frame is None:
        raise RuntimeError("JPEG decompression failed.")
    return frame

if __name__ == '__main__':
    # Example Usage
    dummy_frame_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    print("Testing JPEG compression/decompression:")
    jpeg_quality = 75
    compressed_bvl = compress_frame_jpeg(dummy_frame_color, quality=jpeg_quality)
    print(f"  Original frame shape: {dummy_frame_color.shape}")
    print(f"  Compressed JPEG size (quality={jpeg_quality}): {len(compressed_bvl)} bytes")

    decompressed_bvl = decompress_frame_jpeg(compressed_bvl)
    print(f"  Decompressed frame shape: {decompressed_bvl.shape}")

    # Basic check (won't be identical due to lossy compression)
    if np.allclose(dummy_frame_color, decompressed_bvl, atol=50): # Increased tolerance for JPEG
        print("  Decompression visually similar (basic check passed).")
    else:
        diff = np.sum(np.abs(dummy_frame_color.astype(float) - decompressed_bvl.astype(float)))
        print(f"  Decompression has differences (sum of absolute diff: {diff}). Expected for JPEG.")
    
    assert decompressed_bvl.shape == dummy_frame_color.shape, "Shape mismatch after JPEG cycle."
    print("JPEG utility functions seem operational.")
