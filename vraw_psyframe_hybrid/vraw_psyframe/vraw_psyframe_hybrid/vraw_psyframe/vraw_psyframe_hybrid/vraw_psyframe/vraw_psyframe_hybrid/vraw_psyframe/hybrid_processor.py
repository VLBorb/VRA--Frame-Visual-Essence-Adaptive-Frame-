# vraw_psyframe_hybrid/vraw_psyframe/hybrid_processor.py

from .vraw_encoder import VRAWEncoder
from .chids import CHIDS
from .adaptive_controller import AdaptiveController
from .adaptive_aes import AdaptiveAES
from .utils import compress_frame_jpeg # Using our own utility

import numpy as np
import time

class HybridFrameProcessor:
    """
    Orchestrates the VRA Ψ-Frame hybrid processing for single frames and sequences.
    Combines Base Visual Layer (BVL) with VRAW Essence Layer (VEL)
    and uses an AdaptiveController for interpretation.
    """
    def __init__(self, vraw_freq_pin=None, aes_freq_identity=0.123,
                 controller_alpha=1.0, controller_beta=0.5, controller_gamma=10.0,
                 jpeg_quality=90):
        self.vraw_encoder = VRAWEncoder(freq_pin=vraw_freq_pin)
        self.adaptive_controller = AdaptiveController(
            alpha=controller_alpha, beta=controller_beta, gamma=controller_gamma
        )
        self.aes_cipher = AdaptiveAES(freq_identity=aes_freq_identity)
        self.jpeg_quality = jpeg_quality
        self.chids_store = None # For sequences

    def process_single_frame(self, frame: np.ndarray) -> dict:
        """
        Processes a single frame to generate its hybrid representation.

        :param frame: Input frame as a NumPy array (BGR or Grayscale).
        :return: A dictionary containing:
                 'bvl_jpeg_bytes': JPEG compressed Base Visual Layer.
                 'vel_clear': VRAW Essence Layer parameters (dictionary).
                 'vel_encrypted': Encrypted VEL (bytes).
                 'adaptive_interpretation': Output from AdaptiveController (dictionary).
                 'processing_time_ms': Time taken for processing.
        """
        start_time = time.perf_counter()

        # 1. Base Visual Layer (BVL) - e.g., JPEG compressed
        bvl_jpeg_bytes = compress_frame_jpeg(frame, quality=self.jpeg_quality)

        # 2. VRAW Essence Layer (VEL)
        vel_clear = self.vraw_encoder.encode_frame(frame)

        # 3. Encrypt VEL
        vel_encrypted = self.aes_cipher.encrypt_dict(vel_clear)

        # 4. Adaptive Interpretation using VEL
        adaptive_interpretation = self.adaptive_controller.interpret_vel(vel_clear)
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        return {
            "bvl_jpeg_bytes": bvl_jpeg_bytes,
            "vel_clear": vel_clear,
            "vel_encrypted": vel_encrypted,
            "adaptive_interpretation": adaptive_interpretation,
            "processing_time_ms": processing_time_ms,
        }

    def process_frame_sequence(self, frames: list) -> dict:
        """
        Processes a sequence of frames, stores VELs in CHIDS, and provides
        a temporal summary from the AdaptiveController.

        :param frames: A list of input frames (NumPy arrays).
        :return: A dictionary containing:
                 'processed_frames_data': List of hybrid representations for each frame.
                 'chids_feature_names': List of feature names stored in CHIDS.
                 'chids_data_cube_shape': Shape of the CHIDS data cube.
                 'temporal_vel_summary': Summary from AdaptiveController based on CHIDS data.
                 'total_processing_time_ms': Total time for sequence processing.
        """
        start_time = time.perf_counter()
        
        processed_frames_data = []
        vel_sequence = []

        for frame in frames:
            # For sequence, we might not need to store BVL/encrypted VEL for each frame here
            # Focus on VEL for CHIDS and temporal analysis
            # If individual full processing is needed, call process_single_frame
            
            # Simplified processing for sequence (focus on VEL)
            current_vel = self.vraw_encoder.encode_frame(frame)
            vel_sequence.append(current_vel)
            
            # Store full processing data if needed (can be memory intensive for long sequences)
            # For this example, let's store a simplified version for the list
            processed_frames_data.append({
                "vel_clear": current_vel,
                "adaptive_interpretation": self.adaptive_controller.interpret_vel(current_vel)
            })

        # Build CHIDS store with the collected VEL sequence
        self.chids_store = CHIDS(vel_sequence)
        
        chids_data_cube = self.chids_store.get_data_cube()
        chids_feature_names = self.chids_store.get_feature_names()

        # Get temporal summary from AdaptiveController
        temporal_vel_summary = self.adaptive_controller.process_temporal_vel_summary(
            chids_data_cube,
            chids_feature_names
        )
        
        end_time = time.perf_counter()
        total_processing_time_ms = (end_time - start_time) * 1000

        return {
            "processed_frames_data": processed_frames_data, # Contains VEL and interpretation per frame
            "chids_feature_names": chids_feature_names,
            "chids_data_cube_shape": chids_data_cube.shape if chids_data_cube is not None else None,
            "temporal_vel_summary": temporal_vel_summary,
            "total_processing_time_ms": total_processing_time_ms,
        }

if __name__ == '__main__':
    # Example Usage
    processor = HybridFrameProcessor(jpeg_quality=80, aes_freq_identity=0.25)

    # Create a dummy color frame
    dummy_frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    print("--- Processing Single Frame ---")
    single_frame_result = processor.process_single_frame(dummy_frame)
    
    print(f"  BVL JPEG size: {len(single_frame_result['bvl_jpeg_bytes'])} bytes")
    print(f"  VEL (Clear): {single_frame_result['vel_clear']}")
    print(f"  VEL (Encrypted Sample): {single_frame_result['vel_encrypted'][:16]}... (IV) ...{single_frame_result['vel_encrypted'][16:32]}...")
    print(f"  Adaptive Interpretation: {single_frame_result['adaptive_interpretation']}")
    print(f"  Processing Time: {single_frame_result['processing_time_ms']:.2f} ms")

    # Decrypt VEL to verify
    decrypted_vel = processor.aes_cipher.decrypt_to_dict(single_frame_result['vel_encrypted'])
    assert decrypted_vel == single_frame_result['vel_clear'], "VEL decryption failed!"
    print("  VEL decryption successful.")

    print("\n--- Processing Frame Sequence ---")
    dummy_frames_sequence = []
    for i in range(5):
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        # Make frames slightly different
        frame[:,:,0] = np.clip(frame[:,:,0] + i*10, 0, 255) # Change Blue channel
        frame[:,:,1] = np.clip(frame[:,:,1] - i*5, 0, 255)  # Change Green channel
        dummy_frames_sequence.append(frame)

    sequence_result = processor.process_frame_sequence(dummy_frames_sequence)

    print(f"  Processed {len(sequence_result['processed_frames_data'])} frames.")
    print(f"  CHIDS Feature Names: {sequence_result['chids_feature_names']}")
    print(f"  CHIDS Data Cube Shape: {sequence_result['chids_data_cube_shape']}")
    print(f"  Temporal VEL Summary: {sequence_result['temporal_vel_summary']}")
    print(f"  Total Sequence Processing Time: {sequence_result['total_processing_time_ms']:.2f} ms")
    print(f"  VEL for first frame in sequence: {sequence_result['processed_frames_data'][0]['vel_clear']}")
