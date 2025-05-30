# vraw_psyframe_hybrid/vraw_psyframe/adaptive_controller.py

import numpy as np

class AdaptiveController:
    """
    Uses VRAW Essence Layer (VEL) parameters to adaptively generate
    interpretations or control signals.
    The parameters (alpha, beta, gamma) can be used to tune the sensitivity
    or behavior of the adaptive logic.
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def interpret_vel(self, vel_data: dict) -> dict:
        """
        Interprets a single frame's VRAW Essence Layer (VEL) data.
        This is a sample interpretation logic.

        :param vel_data: A dictionary of VRAW parameters for a single frame.
                         Expected keys: "scale_factor", "offset", "fractal_signature", "edge_density".
        :return: A dictionary containing interpretations or derived metrics.
        """
        interpretations = {}

        # Brightness interpretation based on scale_factor
        sf = vel_data.get("scale_factor", 0.5) # Default to mid-range
        if sf * self.alpha > 0.7:
            interpretations["brightness_assessment"] = "High Brightness"
        elif sf * self.alpha < 0.3:
            interpretations["brightness_assessment"] = "Low Brightness"
        else:
            interpretations["brightness_assessment"] = "Medium Brightness"

        # Contrast interpretation based on offset
        offset = vel_data.get("offset", 0)
        if offset * self.beta > 80: # Tunable threshold
            interpretations["contrast_assessment"] = "High Contrast"
        elif offset * self.beta < 30:
            interpretations["contrast_assessment"] = "Low Contrast"
        else:
            interpretations["contrast_assessment"] = "Medium Contrast"

        # Complexity score (example)
        edge_density = vel_data.get("edge_density", 0)
        complexity_score = (offset * 0.01 + edge_density * 10 + self.gamma)
        interpretations["complexity_score"] = np.clip(complexity_score, 0, 100)

        # Significance based on fractal_signature (example: treat even/odd differently)
        fs = vel_data.get("fractal_signature", 0)
        interpretations["signature_type"] = "Type A (Even)" if fs % 2 == 0 else "Type B (Odd)"
        
        return interpretations

    def process_temporal_vel_summary(self, chids_data_cube: np.ndarray, feature_names: list) -> dict:
        """
        Processes a summary of VEL data over time (e.g., from CHIDS).
        This is a placeholder for more complex temporal analysis.
        For now, it calculates mean and std dev for each feature over time.

        :param chids_data_cube: Numpy array (num_features, num_frames) from CHIDS.
        :param feature_names: List of feature names corresponding to rows in data_cube.
        :return: A dictionary of temporal summary statistics.
        """
        if chids_data_cube is None or chids_data_cube.size == 0:
            return {"temporal_summary_status": "No data provided"}

        summary = {"temporal_summary_status": "Processed"}
        for i, feature_name in enumerate(feature_names):
            feature_vector = chids_data_cube[i, :]
            summary[f"{feature_name}_mean"] = np.mean(feature_vector)
            summary[f"{feature_name}_std"] = np.std(feature_vector)
            if feature_vector.size > 1:
                 change = feature_vector[-1] - feature_vector[0]
                 summary[f"{feature_name}_total_change"] = change
            else:
                 summary[f"{feature_name}_total_change"] = 0
        return summary


if __name__ == '__main__':
    controller = AdaptiveController(alpha=1.0, beta=1.0, gamma=5.0)
    
    sample_vel_1 = {"scale_factor": 0.8, "offset": 90.0, "fractal_signature": 10, "edge_density": 0.25}
    interpretation_1 = controller.interpret_vel(sample_vel_1)
    print("Interpretation for VEL 1:")
    for key, value in interpretation_1.items():
        print(f"  {key}: {value}")

    sample_vel_2 = {"scale_factor": 0.2, "offset": 20.0, "fractal_signature": 7, "edge_density": 0.05}
    interpretation_2 = controller.interpret_vel(sample_vel_2)
    print("\nInterpretation for VEL 2:")
    for key, value in interpretation_2.items():
        print(f"  {key}: {value}")

    # Example for temporal processing
    from chids import CHIDS # Assuming chids.py is in the same directory for direct run
    sample_vel_sequence = [
        {"scale_factor": 0.5, "offset": 70.0, "fractal_signature": 10, "edge_density": 0.1},
        {"scale_factor": 0.6, "offset": 72.0, "fractal_signature": 15, "edge_density": 0.12},
        {"scale_factor": 0.4, "offset": 68.0, "fractal_signature": 5, "edge_density": 0.09},
    ]
    chids_store = CHIDS(sample_vel_sequence)
    temporal_summary = controller.process_temporal_vel_summary(
        chids_store.get_data_cube(), 
        chids_store.get_feature_names()
    )
    print("\nTemporal VEL Summary:")
    for key, value in temporal_summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
