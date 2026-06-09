ARG OLLAMA_VERSION=0.30.7
# Optional llama.cpp git tag or SHA.
# - Modern Ollama (has LLAMA_CPP_VERSION): defaults to that file's tag (e.g. b9509 for v0.30.7).
# - Legacy Ollama (no LLAMA_CPP_VERSION): defaults to the pin in the RUN script if unset.
ARG GGML_COMMIT
ARG COMPUTE_RUNTIME_VERSION=26.14.37833.4
ARG LEVEL_ZERO_VERSION=1.28.2
ARG IGC_VERSION=2.32.7
ARG IGC_BUILD=21184
ARG GMM_VERSION=22.9.0

# Default: oneAPI Base Kit 2025.2.x.
# For Intel® oneAPI Toolkit 2026 (no SYCLcompat headers), build with:
#   --build-arg ONEAPI_TOOLKIT_IMAGE=intel/oneapi-toolkit:2026.0.0-devel-ubuntu24.04 \
#   --build-arg STRIP_SYCLCOMPAT=1
ARG ONEAPI_TOOLKIT_IMAGE=intel/oneapi-basekit:2025.2.2-0-devel-ubuntu24.04

FROM ${ONEAPI_TOOLKIT_IMAGE} AS sycl-builder

ARG STRIP_SYCLCOMPAT=0
ARG OLLAMA_VERSION
ARG GGML_COMMIT

# Ollama >= ~0.29: ggml is no longer fully vendored under ml/backend/ggml; the repo ships
# LLAMA_CPP_VERSION and builds via FetchContent from llama/ (see root CMakeLists.txt).
# Ollama < 0.29: shallow clone still contains ml/backend/ggml/... and we inject ggml-sycl there.
RUN git clone --depth 1 --branch "v${OLLAMA_VERSION}" \
  https://github.com/ollama/ollama.git /ollama

COPY patch-sycl.py /tmp/patch-sycl.py

RUN set -eux; \
  export GIT_TERMINAL_PROMPT=0; \
  if [ -f /ollama/LLAMA_CPP_VERSION ]; then \
    echo "[ollama-sycl] modern layout: standalone llama.cpp from LLAMA_CPP_VERSION"; \
    if [ -n "${GGML_COMMIT:-}" ]; then LREF="${GGML_COMMIT}"; \
    else LREF="$(tr -d ' \t\r\n' < /ollama/LLAMA_CPP_VERSION)"; fi; \
    echo "[ollama-sycl] llama.cpp ref=${LREF}"; \
    rm -rf /work/llama.cpp; \
    mkdir -p /work/llama.cpp; \
    git -C /work/llama.cpp init; \
    git -C /work/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git; \
    git -C /work/llama.cpp fetch --depth 1 origin "${LREF}"; \
    git -C /work/llama.cpp checkout -q FETCH_HEAD; \
    test -f /work/llama.cpp/ggml/CMakeLists.txt; \
    cd /work/llama.cpp && cmake -D PATCH_DIR=/ollama/llama/compat -P /ollama/llama/compat/apply-patch.cmake; \
    SYCL_CPP="$(find /work/llama.cpp -type f -path '*/ggml-sycl/ggml-sycl.cpp' | head -1)"; \
    ABI_HDR="$(find /work/llama.cpp -type f -name 'ggml-backend-impl.h' | head -1)"; \
    echo "[patch-sycl] SYCL_CPP=${SYCL_CPP}"; \
    echo "[patch-sycl] ABI_HDR=${ABI_HDR}"; \
    test -n "${SYCL_CPP}" && test -n "${ABI_HDR}"; \
    python3 /tmp/patch-sycl.py "${SYCL_CPP}" --abi-header "${ABI_HDR}"; \
    if [ "${STRIP_SYCLCOMPAT}" = "1" ]; then \
      HELPER="$(find /work/llama.cpp /ollama -type f -path '*/ggml-sycl/dpct/helper.hpp' 2>/dev/null | head -1)"; \
      echo "[patch-sycl] SYCLCOMPAT helper=${HELPER:-MISSING}"; \
      if [ -n "${HELPER}" ]; then sed -i '\|#include <syclcompat/math.hpp>|d' "${HELPER}"; fi; \
    fi; \
    cmake -S /work/llama.cpp -B /work/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON \
      -DGGML_SYCL=ON \
      -DGGML_SYCL_TARGET=INTEL \
      -DGGML_SYCL_F16=ON; \
    cmake --build /work/llama.cpp/build --parallel "$(nproc)" --target ggml-sycl; \
  else \
    echo "[ollama-sycl] legacy layout: inject ggml-sycl under ml/backend/ggml"; \
    PIN="${GGML_COMMIT:-15bff84bf56651d6f991f166a2bf0f362996f7f9}"; \
    git init /tmp/llama.cpp && cd /tmp/llama.cpp && \
      git remote add origin https://github.com/ggml-org/llama.cpp.git && \
      git sparse-checkout set ggml/src/ggml-sycl && \
      git fetch --depth 1 origin "${PIN}" && \
      git checkout FETCH_HEAD && \
      cp -r ggml/src/ggml-sycl /ollama/ml/backend/ggml/ggml/src/ && \
      cd / && rm -rf /tmp/llama.cpp; \
    SYCL_CPP="$(find /ollama -type f -path '*/ggml-sycl/ggml-sycl.cpp' | head -1)"; \
    ABI_HDR="$(find /ollama -type f -name 'ggml-backend-impl.h' | head -1)"; \
    echo "[patch-sycl] SYCL_CPP=${SYCL_CPP}"; \
    echo "[patch-sycl] ABI_HDR=${ABI_HDR}"; \
    test -n "${SYCL_CPP}" && test -n "${ABI_HDR}"; \
    python3 /tmp/patch-sycl.py "${SYCL_CPP}" --abi-header "${ABI_HDR}"; \
    if [ "${STRIP_SYCLCOMPAT}" = "1" ]; then \
      HELPER="$(find /ollama -type f -path '*/ggml-sycl/dpct/helper.hpp' | head -1)"; \
      echo "[patch-sycl] SYCLCOMPAT helper=${HELPER:-MISSING}"; \
      if [ -n "${HELPER}" ]; then sed -i '\|#include <syclcompat/math.hpp>|d' "${HELPER}"; fi; \
    fi; \
    cmake -S /ollama -B /ollama/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_SYCL=ON \
      -DGGML_SYCL_TARGET=INTEL \
      -DGGML_SYCL_F16=ON \
      -DOLLAMA_RUNNER_DIR=sycl; \
    cmake --build /ollama/build --parallel "$(nproc)" --target ggml-sycl; \
  fi

# Collect the SYCL runner and its oneAPI runtime dependencies into /sycl-runner
RUN mkdir -p /sycl-runner && \
  SO="$(find /ollama/build /work/llama.cpp/build -name 'libggml-sycl.so' 2>/dev/null | head -1)" && \
  test -n "${SO}" && echo "[ollama-sycl] bundling ${SO}" && cp "${SO}" /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl.so* /sycl-runner/ && \
  find /opt/intel/oneapi -name 'libur_loader.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  find /opt/intel/oneapi -name 'libur_adapter_level_zero.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  find /opt/intel/oneapi -maxdepth 4 -name 'libumf.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  cp /opt/intel/oneapi/dnnl/latest/lib/libdnnl.so* /sycl-runner/ 2>/dev/null; \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_intel_ilp64.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_tbb_thread.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/tbb/latest/lib/intel64/gcc*/libtbb.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libsvml.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libimf.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libintlc.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libirng.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libiomp5.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libpi_level_zero.so* /sycl-runner/ 2>/dev/null; \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-fallback*.spv /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-native*.spv /sycl-runner/ && \
  strip --strip-unneeded /sycl-runner/*.so* 2>/dev/null; true

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

ARG OLLAMA_VERSION
ARG COMPUTE_RUNTIME_VERSION
ARG LEVEL_ZERO_VERSION
ARG IGC_VERSION
ARG IGC_BUILD
ARG GMM_VERSION

# Base packages
RUN apt-get update && \
  apt-get install --no-install-recommends -q -y \
  ca-certificates \
  wget \
  zstd \
  ocl-icd-libopencl1 \
  libhwloc15 && \
  rm -rf /var/lib/apt/lists/*

# Intel GPU runtimes
RUN mkdir -p /tmp/gpu && cd /tmp/gpu && \
  wget https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMM_VERSION}_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
  dpkg -i *.deb && rm -rf /tmp/gpu

# Install official ollama binary + CPU runners
RUN wget -qO- "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | \
  zstd -d | tar -xf - -C /usr && \
  rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/mlx_* /usr/lib/ollama/vulkan

# Install SYCL runner from build stage
COPY --from=sycl-builder /sycl-runner/ /usr/lib/ollama/sycl/

ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_KEEP_ALIVE=24h
ENV OLLAMA_FLASH_ATTENTION=false
ENV ZES_ENABLE_SYSMAN=1
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0
# Optional: Intel Core Ultra (Series 1) Meteor Lake integrated Arc
ENV IPEX_LLM_NPU_MTL=1

EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
