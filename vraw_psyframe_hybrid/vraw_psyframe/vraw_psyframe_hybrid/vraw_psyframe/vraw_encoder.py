# vraw_psyframe_hybrid/vraw_psyframe/vraw_encoder.py

import numpy as np
import cv2

class VRAWEncoder:
    """
    Video encoder that computes:
      - scale_factor (average pixel value / 255)
      - offset (std dev of pixel values)
      - fractal_signature (sum of pixel values mod 256)
      - edge_density (proportion of edge pixels after Canny detection)

    freq_pin: FPIS Dynamic Frequency Pin (currently conceptual, not used in encoding logic)
    """
    def __init__(self, freq_pin=None, canny_threshold1=50, canny_threshold2=150):
        self.freq_pin = freq_pin # Conceptual
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

    def encode_frame(self, frame: np.ndarray):
        """
        Encodes a single frame to extract VRAW parameters.

        :param frame: A single 2D (grayscale) or 3D (color) numpy array.
                      If color, it will be converted to grayscale for some metrics.
        :return: dict of VRAW parameters (scale_factor, offset, fractal_signature, edge_density)
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")
        if frame.ndim == 3 and frame.shape[2] == 3: # Color image
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2: # Grayscale image
            gray_frame = frame
        else:
            raise ValueError("Frame must be a 2D (grayscale) or 3D (color BGR) numpy array.")

        if gray_frame.size == 0:
            raise ValueError("Frame contains no pixels.")

        scale_factor = np.mean(gray_frame) / 255.0
        offset = np.std(gray_frame)
        fractal_signature = int(np.sum(gray_frame)) % 256 # Retains original definition

        # Edge Density Calculation
        edges = cv2.Canny(gray_frame, self.canny_threshold1, self.canny_threshold2)
        edge_density = np.sum(edges > 0) / (gray_frame.size if gray_frame.size > 0 else 1)

        return {
            "scale_factor": scale_factor,
            "offset": offset,
            "fractal_signature": fractal_signature,
            "edge_density": edge_density,
        }

    def batch_encode_frames(self, frames: list):
        """
        Encodes a list of frames.
        :param frames: A list of 2D (grayscale) or 3D (color) numpy arrays.
        :return: A list of dicts, where each dict contains VRAW parameters for a frame.
        """
        encoded_params_list = []
        for frame in frames:
            encoded_params_list.append(self.encode_frame(frame))
        return encoded_params_list

if __name__ == '__main__':
    # Example Usage
    dummy_frame_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    dummy_frame_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    encoder = VRAWEncoder()

    print("Encoding Color Frame:")
    params_color = encoder.encode_frame(dummy_frame_color)
    for key, value in params_color.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


    print("\nEncoding Grayscale Frame:")
    params_gray = encoder.encode_frame(dummy_frame_gray)
    for key, value in params_gray.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Batch encoding example
    batch_frames = [dummy_frame_color, dummy_frame_gray]
    batch_params = encoder.batch_encode_frames(batch_frames)
    print(f"\nBatch encoded {len(batch_params)} frames.")
    print(f"Params for first frame in batch: {batch_params[0]}")
