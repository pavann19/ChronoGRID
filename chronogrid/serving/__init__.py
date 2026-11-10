"""Serving exports, loaded lazily so inference does not require conversion tooling."""

__all__ = ["export_to_onnx", "OnnxInferenceEngine"]


def __getattr__(name):
    if name == "export_to_onnx":
        from chronogrid.serving.onnx_exporter import export_to_onnx

        return export_to_onnx
    if name == "OnnxInferenceEngine":
        from chronogrid.serving.onnx_engine import OnnxInferenceEngine

        return OnnxInferenceEngine
    raise AttributeError(name)
