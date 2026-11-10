"""
High-Performance ONNX Runtime Inference Engine (P5.8).
Loads ONNX model and executes ultra-low latency predictions.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

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
        self.onnx_model_path: Path | None = Path(onnx_model_path)
        self.metadata_path: Path | None = Path(metadata_path)
        self.session: Any = None
        self.metadata: dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.input_name: str = "float_input"
        self.output_name: str = "variable"
        self._load()

    def _load(self):
        """Loads ONNX runtime session and metadata."""
        if self.onnx_model_path is None or not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path}")

        if self.metadata_path is None or not self.metadata_path.exists():
            raise FileNotFoundError(f"Model feature metadata not found at {self.metadata_path}")

        self._load_bytes(self.onnx_model_path.read_bytes(), self.metadata_path.read_bytes())

    @classmethod
    def from_bytes(cls, model_bytes: bytes, metadata_bytes: bytes):
        """Load one verified registry snapshot without reopening mutable file paths."""
        engine = cls.__new__(cls)
        engine.onnx_model_path = None
        engine.metadata_path = None
        engine._load_bytes(model_bytes, metadata_bytes)
        return engine

    def _load_bytes(self, model_bytes, metadata_bytes):
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        self.metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
        self.metadata = json.loads(metadata_bytes)
        if not isinstance(self.metadata, dict):
            raise ValueError("Model metadata must be a JSON object.")

        # Optimize ONNX Runtime session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.feature_names = self.metadata.get("feature_names", [])
        shape = self.session.get_inputs()[0].shape
        if (
            not isinstance(self.feature_names, list)
            or not self.feature_names
            or any(not isinstance(name, str) or not name for name in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Model metadata must contain unique non-empty feature names.")
        if len(shape) != 2 or not isinstance(shape[1], int) or len(self.feature_names) != shape[1]:
            raise ValueError("Model metadata feature count does not match ONNX input shape.")

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
            if not features:
                raise ValueError("At least one feature row is required.")
            required = set(self.feature_names)
            if any(set(row) != required for row in features):
                raise ValueError(
                    "Exact model feature schema required; missing or extra fields are not zero-filled."
                )
            matrix: Any = np.empty((len(features), len(self.feature_names)), dtype=np.float32)
            for i, row in enumerate(features):
                for j, f in enumerate(self.feature_names):
                    matrix[i, j] = float(row[f])
            X = matrix
        else:
            X = np.asarray(features, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        expected_width = len(self.feature_names) or self.session.get_inputs()[0].shape[-1]
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != expected_width:
            raise ValueError(f"Expected non-empty (N, {expected_width}) feature matrix.")
        if not np.isfinite(X).all():
            raise ValueError("Model features must be finite.")

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: X})
        preds = np.asarray(outputs[0]).flatten()
        if len(preds) != X.shape[0] or not np.isfinite(preds).all():
            raise ValueError("Invalid model output shape or non-finite predictions.")
        return preds
