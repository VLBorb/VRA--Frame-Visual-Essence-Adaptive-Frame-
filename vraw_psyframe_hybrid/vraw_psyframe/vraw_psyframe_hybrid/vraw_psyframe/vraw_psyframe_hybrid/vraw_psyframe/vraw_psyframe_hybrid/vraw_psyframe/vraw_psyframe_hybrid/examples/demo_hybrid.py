# vraw_psyframe_hybrid/examples/demo_hybrid.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Adjust import path if running from root of project vs. examples directory
try:
    from vraw_psyframe import (HybridFrameProcessor, VRAWEncoder, CHIDS, 
                               AdaptiveController, AdaptiveAES, 
                               decompress_frame_jpeg)
except ImportError:
    import sys
    # Add the parent directory (vraw_psyframe_hybrid) to sys.path
    # This allows finding the vraw_psyframe package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from vraw_psyframe import (HybridFrameProcessor, VRAWEncoder, CHIDS,
                               AdaptiveController, AdaptiveAES,
                               decompress_frame_jpeg)


def generate_sample_frames(num_frames=5, shape=(128, 128, 3)):
    """Generates a list of somewhat varying random frames."""
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, shape, dtype=np.uint8)
        # Add some variation
        cv2.putText(frame, f"F{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, int(i*255/num_frames)), 2)
        frame[:, :, i % 3] = np.clip(frame[:, :, i % 3] + i * 20, 0, 255)
        frames.append(frame)
    return frames

def display_frame_with_info(bvl_jpeg_bytes, vel_clear, adaptive_interpretation, title="Frame"):
    """Helper to display a frame and its info using matplotlib."""
    try:
        frame = decompress_frame_jpeg(bvl_jpeg_bytes)
        if frame is None:
            print("Failed to decompress BVL for display.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if frame.ndim == 3:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(frame, cmap='gray')
        
        ax.set_title(title)
        ax.axis('off')

        info_text = "VEL:\n"
        for k, v in vel_clear.items():
            info_text += f"  {k}: {v:.3f}\n" if isinstance(v, float) else f"  {k}: {v}\n"
        
        info_text += "\nInterpretation:\n"
        for k, v in adaptive_interpretation.items():
            info_text += f"  {k}: {v}\n"

        fig.text(0.05, 0.05, info_text, fontsize=8, wrap=True,
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        plt.tight_layout(rect=[0, 0.2, 1, 1]) # Adjust layout to make space for text
        plt.show()
    except Exception as e:
        print(f"Error displaying frame: {e}")


def plot_temporal_features(chids_data_cube, feature_names, title="Temporal Feature Evolution"):
    """Plots features over time from CHIDS data."""
    if chids_data_cube is None or chids_data_cube.size == 0:
        print("No CHIDS data to plot.")
        return
        
    num_features, num_frames = chids_data_cube.shape
    if num_frames <= 1:
        print("Not enough frames in CHIDS data to plot temporal evolution.")
        return

    time_indices = np.arange(num_frames)
    
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True)
    if num_features == 1: # Matplotlib returns a single Axes object if nrows=1
        axes = [axes] 

    for i, feature_name in enumerate(feature_names):
        axes[i].plot(time_indices, chids_data_cube[i, :], marker='o', linestyle='-')
        axes[i].set_ylabel(feature_name)
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Frame Index")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()


def main_demo():
    print("=== VRA Ψ-Frame Hybrid System Demo ===")

    # --- Configuration ---
    num_sequence_frames = 7
    frame_shape = (128, 128, 3) # H, W, C
    aes_seed = 0.6789
    jpeg_q = 75

    processor = HybridFrameProcessor(
        aes_freq_identity=aes_seed,
        jpeg_quality=jpeg_q,
        controller_alpha=1.1, # Tweak controller params for different behavior
        controller_beta=0.6
    )

    # --- 1. Single Frame Processing Demo ---
    print("\n[1. Single Frame Processing]")
    sample_frames_for_single = generate_sample_frames(num_frames=1, shape=frame_shape)
    single_frame = sample_frames_for_single[0]

    if single_frame is not None:
        print(f"Processing a single frame of shape {single_frame.shape}...")
        single_result = processor.process_single_frame(single_frame)
        
        print(f"  BVL JPEG size: {len(single_result['bvl_jpeg_bytes'])} bytes")
        print(f"  VEL (Clear):")
        for k,v in single_result['vel_clear'].items(): print(f"    {k}: {v}")
        # print(f"  VEL (Encrypted Sample): {single_result['vel_encrypted'][:16]}...")
        print(f"  Adaptive Interpretation:")
        for k,v in single_result['adaptive_interpretation'].items(): print(f"    {k}: {v}")
        print(f"  Processing Time: {single_result['processing_time_ms']:.2f} ms")

        # Decrypt and verify
        decrypted_vel = processor.aes_cipher.decrypt_to_dict(single_result['vel_encrypted'])
        if decrypted_vel == single_result['vel_clear']:
            print("  VEL encryption/decryption cycle successful.")
        else:
            print("  ERROR: VEL encryption/decryption mismatch!")
        
        # Display the frame and its info
        display_frame_with_info(
            single_result['bvl_jpeg_bytes'],
            single_result['vel_clear'],
            single_result['adaptive_interpretation'],
            title="Single Processed Frame"
        )
    else:
        print("Could not generate single sample frame.")

    input("Press Enter to continue to sequence processing...")

    # --- 2. Frame Sequence Processing Demo ---
    print("\n[2. Frame Sequence Processing]")
    sequence_frames = generate_sample_frames(num_frames=num_sequence_frames, shape=frame_shape)
    
    if sequence_frames:
        print(f"Processing a sequence of {len(sequence_frames)} frames...")
        sequence_result = processor.process_frame_sequence(sequence_frames)

        print(f"  Total Sequence Processing Time: {sequence_result['total_processing_time_ms']:.2f} ms")
        print(f"  CHIDS Feature Names: {sequence_result['chids_feature_names']}")
        print(f"  CHIDS Data Cube Shape: {sequence_result['chids_data_cube_shape']}")
        
        print("\n  Temporal VEL Summary (from AdaptiveController):")
        for k, v in sequence_result['temporal_vel_summary'].items():
            print(f"    {k}: {v:.4f}" if isinstance(v, (float, np.float64)) else f"    {k}: {v}")
        
        # Example: Accessing VEL for a specific frame from the sequence result
        if sequence_result['processed_frames_data']:
             print(f"\n  VEL for Frame 0 in sequence: {sequence_result['processed_frames_data'][0]['vel_clear']}")

        # Plot temporal features from CHIDS
        chids_module = CHIDS(vel_sequence=[d['vel_clear'] for d in sequence_result['processed_frames_data']]) # Rebuild for plot
        plot_temporal_features(
            chids_module.get_data_cube(),
            chids_module.get_feature_names(),
            title=f"Temporal Evolution of VEL ({num_sequence_frames} Frames)"
        )
    else:
        print("Could not generate frame sequence.")

    print("\nDemo finished.")

if __name__ == "__main__":
    main_demo()
