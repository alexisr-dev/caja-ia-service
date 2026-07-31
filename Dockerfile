# Se fija la release de Debian a proposito. La etiqueta flotante python:3.11-slim
# migro a Debian 13 (trixie), donde libgl1-mesa-glx ya no existe y libglib2.0-0
# se renombro a libglib2.0-0t64: el build se rompio sin cambiar una sola linea.
FROM python:3.11-slim-bookworm

# ULTRALYTICS_AUTOINSTALL=false: impide que ultralytics ejecute pip en caliente.
ENV ULTRALYTICS_AUTOINSTALL=false \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgl1/libglib2: dependencias nativas de opencv (lo arrastra ultralytics).
# git: necesario para instalar CLIP, que no está publicado en PyPI.
# curl: usado por el HEALTHCHECK.
# Nombres validos en bookworm. Si algun dia se sube a trixie, libglib2.0-0 pasa
# a llamarse libglib2.0-0t64 (transicion time_t de 64 bits).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Usuario sin privilegios. El volumen de la base vectorial debe ser escribible
# por este uid (montarlo con el owner correcto, ver README).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/db_vectorial \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
