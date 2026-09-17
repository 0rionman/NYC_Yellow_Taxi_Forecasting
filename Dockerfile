ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN if [ -f requirements.lock.txt ]; then pip install --no-cache-dir -r requirements.lock.txt; else pip install --no-cache-dir -r requirements.txt; fi && pip check && pip freeze > requirements.lock.txt
COPY taxi_project ./taxi_project
COPY app.py ./app.py
RUN useradd -m taxi && mkdir /app/artifacts && chown -R taxi:taxi /app
USER taxi
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
