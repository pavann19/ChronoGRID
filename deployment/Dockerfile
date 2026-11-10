# Locked runtime install; compatible with classic Docker and BuildKit.
FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1 AS locked-runtime
COPY deployment/requirements-service.lock /requirements.lock
RUN python -m venv /opt/venv && /opt/venv/bin/python -m pip install --no-cache-dir --require-hashes -r /requirements.lock && /opt/venv/bin/python -m pip check && /opt/venv/bin/python -m pip uninstall --yes pip
FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    DATABASE_URL=sqlite:////state/registry.sqlite \
    DDIR_STATE_FILE=/state/ddir_state.json DQCP_STATE_FILE=/state/dqcp_aci_state.json
WORKDIR /app
RUN python -m pip uninstall --yes pip && rm -rf /usr/local/lib/python3.13/ensurepip
COPY --from=locked-runtime /opt/venv/ /opt/venv/
ENV PATH="/opt/venv/bin:$PATH"
COPY chronogrid/ /app/chronogrid/
COPY artifacts/models/ /app/artifacts/models/
COPY artifacts/calibration/ /app/artifacts/calibration/
RUN useradd -u 10001 -m chronogrid && mkdir /state && chown chronogrid:chronogrid /state
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["uvicorn","chronogrid.serving.app:app","--host","0.0.0.0","--port","8000","--workers","1"]
