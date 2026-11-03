import importlib

packages = [
    "pandas", "numpy", "scipy", "sklearn", "xgboost", "lightgbm",
    "mlflow", "fastapi", "uvicorn", "pydantic", "evidently",
    "prometheus_client", "onnx", "onnxruntime", "skl2onnx", "onnxmltools",
    "sqlalchemy", "psycopg2", "asyncpg", "httpx", "requests", "joblib"
]

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"[OK] {pkg}: {ver}")
    except Exception as e:
        print(f"[MISSING] {pkg}: {e}")
