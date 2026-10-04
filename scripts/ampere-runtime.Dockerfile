# Build from the prepared build/ampere directory (containing bundle/).
FROM ubuntu:24.04
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends python3 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ampere
COPY bundle/ ./
ENTRYPOINT ["python3", "run_cuda_ampere.py"]
CMD ["--output", "/results", "--budget-seconds", "2700"]
