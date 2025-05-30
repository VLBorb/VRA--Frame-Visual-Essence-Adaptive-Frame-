# vraw_psyframe_hybrid/vraw_psyframe/chids.py

import numpy as np

class CHIDS:
    """
    Cubic Holographic Indexed Data Storage.
    Stores features (or other data) extracted from frames in a structured way.
    Primarily designed to work with the VRAW Essence Layer (VEL) parameters.
    """
    def __init__(self, vel_sequence: list = None):
        """
        Initializes CHIDS with a sequence of VRAW Essence Layer parameters.
        Each element in vel_sequence is expected to be a dictionary of features.

        :param vel_sequence: A list of dictionaries, where each dictionary
                             represents the VEL for a frame.
        """
        self.feature_names = []
        self.data_cube = np.array([]) # Shape: (num_features, num_frames)

        if vel_sequence:
            self.build_from_vel_sequence(vel_sequence)

    def build_from_vel_sequence(self, vel_sequence: list):
        """
        Builds the internal data cube from a list of VEL dictionaries.
        Assumes all dictionaries in the sequence have the same keys (features).
        """
        if not vel_sequence:
            self.feature_names = []
            self.data_cube = np.array([])
            return

        # Determine feature names from the first VEL entry
        self.feature_names = sorted(list(vel_sequence[0].keys()))
        num_features = len(self.feature_names)
        num_frames = len(vel_sequence)

        self.data_cube = np.zeros((num_features, num_frames), dtype=np.float64)

        for frame_idx, vel_data in enumerate(vel_sequence):
            for feature_idx, feature_name in enumerate(self.feature_names):
                self.data_cube[feature_idx, frame_idx] = vel_data.get(feature_name, np.nan)

    def get_feature_names(self) -> list:
        return self.feature_names

    def get_data_cube(self) -> np.ndarray:
        """Returns the data cube (num_features, num_frames)."""
        return self.data_cube

    def get_temporal_feature_vector(self, feature_name: str) -> np.ndarray:
        """
        Extracts the temporal vector for a specific feature.
        :param feature_name: The name of the feature to extract.
        :return: A 1D numpy array of the feature's values over time, or None if not found.
        """
        if feature_name not in self.feature_names:
            # print(f"Warning: Feature '{feature_name}' not found in CHIDS.")
            return None
        feature_idx = self.feature_names.index(feature_name)
        return self.data_cube[feature_idx, :]

    def get_frame_features(self, frame_index: int) -> dict:
        """
        Retrieves all features for a specific frame index.
        :param frame_index: The index of the frame.
        :return: A dictionary of features for that frame, or None if index is out of bounds.
        """
        if not (0 <= frame_index < self.data_cube.shape[1]):
            # print(f"Warning: Frame index {frame_index} is out of bounds.")
            return None
        
        frame_data = {}
        for feature_idx, feature_name in enumerate(self.feature_names):
            frame_data[feature_name] = self.data_cube[feature_idx, frame_index]
        return frame_data

    # The original 'diagonal_access' was for pixel data.
    # For feature data, a direct "diagonal" might not be as meaningful
    # unless the cube shape is (feature_dim1, feature_dim2, num_frames).
    # Here, data_cube is (num_features, num_frames).
    # A "diagonal" could mean a specific combination if num_features == num_frames,
    # or a diagonal across a reshaped representation.
    # For now, we'll provide access to features over time or features per frame.

if __name__ == '__main__':
    # Example Usage:
    sample_vel_sequence = [
        {"scale_factor": 0.5, "offset": 70.0, "fractal_signature": 10, "edge_density": 0.1},
        {"scale_factor": 0.6, "offset": 72.0, "fractal_signature": 15, "edge_density": 0.12},
        {"scale_factor": 0.4, "offset": 68.0, "fractal_signature": 5, "edge_density": 0.09},
        {"scale_factor": 0.55, "offset": 71.0, "fractal_signature": 12, "edge_density": 0.11},
    ]

    chids_store = CHIDS(sample_vel_sequence)

    print("Feature Names:", chids_store.get_feature_names())
    print("\nData Cube (num_features, num_frames):")
    print(chids_store.get_data_cube())

    print("\nTemporal vector for 'offset':")
    print(chids_store.get_temporal_feature_vector("offset"))

    print("\nFeatures for frame 2:")
    print(chids_store.get_frame_features(2))

    print("\nFeatures for non-existent feature 'brightness':")
    print(chids_store.get_temporal_feature_vector("brightness"))
