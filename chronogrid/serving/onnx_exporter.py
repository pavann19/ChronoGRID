"""
ONNX Model Exporter (P5.8).
Converts LightGBM, XGBoost, and Scikit-learn models into standard ONNX format.
"""

import json
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_wrapper: Any,
    feature_names: List[str],
    output_path: str = "artifacts/models/champion_model.onnx",
    metadata_path: str = "artifacts/models/champion_metadata.json",
) -> str:
    """
    Exports trained model wrapper to ONNX format and saves feature metadata.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_features = len(feature_names)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    underlying_model = getattr(model_wrapper, "model", model_wrapper)
    model_name = getattr(model_wrapper, "name", type(underlying_model).__name__)

    onnx_model = None

    # 1. LightGBM conversion
    if "LightGBM" in model_name or "LGBM" in type(underlying_model).__name__:
        try:
            import onnxmltools

            onnx_model = onnxmltools.convert_lightgbm(
                underlying_model, initial_types=initial_type, target_opset=15
            )
            logger.info("Successfully converted LightGBM model via onnxmltools.")
        except Exception as e:
            logger.warning(f"onnxmltools conversion failed ({e}), trying skl2onnx...")

    # 2. XGBoost conversion
    elif "XGB" in model_name or "XGB" in type(underlying_model).__name__:
        try:
            import onnxmltools

            # onnxmltools requires XGBoost booster feature names to follow f0, f1... pattern
            assert underlying_model is not None
            booster = underlying_model.get_booster()
            original_feature_names = booster.feature_names
            booster.feature_names = [f"f{i}" for i in range(n_features)]
            onnx_model = onnxmltools.convert_xgboost(
                underlying_model, initial_types=initial_type, target_opset=15
            )
            # Restore original feature names
            booster.feature_names = original_feature_names
            logger.info("Successfully converted XGBoost model via onnxmltools.")
        except Exception as e:
            logger.warning(f"onnxmltools XGBoost conversion failed ({e}), trying skl2onnx...")

    # 3. Scikit-learn / General conversion
    if onnx_model is None:
        try:
            # Patch onnx make_attribute for skl2onnx boolean array compatibility in newer ONNX versions
            orig_make_attr = onnx.helper.make_attribute

            def safe_make_attribute(key, value, doc_string=None):
                if isinstance(value, (list, tuple)) and any(
                    isinstance(x, (bool, np.bool_)) for x in value
                ):
                    value = [int(x) if isinstance(x, (bool, np.bool_)) else x for x in value]
                return orig_make_attr(key, value, doc_string=doc_string)

            onnx.helper.make_attribute = safe_make_attribute

            from skl2onnx import convert_sklearn

            onnx_model = convert_sklearn(
                underlying_model, initial_types=initial_type, target_opset=15
            )
            logger.info("Successfully converted model via skl2onnx.")
        except Exception as e:
            logger.error(f"skl2onnx conversion failed: {e}")
            raise
        finally:
            onnx.helper.make_attribute = orig_make_attr

    # Save ONNX binary
    with open(out_file, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Save Metadata JSON
    meta = {
        "model_type": model_name,
        "n_features": n_features,
        "feature_names": feature_names,
        "input_name": "float_input",
        "output_name": onnx_model.graph.output[0].name,
        "onnx_path": str(out_file),
    }
    with open(metadata_path, "w") as metadata_stream:
        json.dump(meta, metadata_stream, indent=2)

    logger.info(f"Model exported to {out_file} (Input dim: {n_features})")
    return str(out_file)
