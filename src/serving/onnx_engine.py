"""
High-Performance ONNX Runtime Inference Engine (P5.8).
Loads ONNX model and executes ultra-low latency predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class OnnxInferenceEngine:
    """
    Sub-millisecond inference engine using ONNX Runtime.
    """
    def __init__(
        self,
        onnx_model_path: str = "artifacts/models/champion_model.onnx",
        metadata_path: str = "artifacts/models/champion_metadata.json",
    ):
        self.onnx_model_path = Path(onnx_model_path)
        self.metadata_path = Path(metadata_path)
        self.session = None
        self.metadata = {}
        self.feature_names: List[str] = []
        self.input_name: str = "float_input"
        self.output_name: str = "variable"
        self._load()

    def _load(self):
        """Loads ONNX runtime session and metadata."""
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path}")

        # Optimize ONNX Runtime session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get("feature_names", [])

        logger.info(
            f"ONNX Inference Engine initialized. Input: {self.input_name}, "
            f"Output: {self.output_name}, Features: {len(self.feature_names)}"
        )

    def predict(self, features: Union[np.ndarray, List[Dict[str, float]]]) -> np.ndarray:
        """
        Executes ONNX model prediction.
        features: numpy array shape (N, D) or list of dicts matching feature_names.
        """
        if isinstance(features, list):
            # Convert list of dicts to 2D numpy float32 array
            if not self.feature_names:
                raise ValueError("Metadata feature names required when passing list of dicts.")
            matrix = np.empty((len(features), len(self.feature_names)), dtype=np.float32)
            for i, row in enumerate(features):
                for j, f in enumerate(self.feature_names):
                    matrix[i, j] = float(row.get(f, 0.0))
            X = matrix
        else:
            X = np.asarray(features, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: X})
        preds = np.asarray(outputs[0]).flatten()
        return preds

