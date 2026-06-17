# parakeet.cpp container image.
#
# Multi-stage build: a fat build stage compiles parakeet-cli and
# parakeet-server (and the ggml backends they link against), then slim runtime
# stages carry only one binary plus the ggml shared libraries. Two runtime
# targets are exposed:
#   --target runtime         the cli image (default)
#   --target runtime-server  the OpenAI-compatible HTTP server image
#
# The same Dockerfile produces the CPU and CUDA variants. Select with build
# args:
#
#   CPU (default):
#     docker build -t parakeet.cpp:cpu .
#
#   CUDA (GGML_CUDA_NO_VMM=ON drops the libcuda driver-lib link dependency,
#   which a GPU-less build container does not have):
#     docker build -t parakeet.cpp:cuda \
#       --build-arg BUILD_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04 \
#       --build-arg "CMAKE_EXTRA_ARGS=-DPARAKEET_GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON" .
#
# The build context must be a checkout with the ggml submodule populated
# (git clone --recursive, or actions/checkout with submodules: recursive).
# Models are not bundled: mount a pre-converted .gguf at runtime.

ARG BUILD_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

# ---------------------------------------------------------------------------
# build: configure + compile parakeet-cli and the ggml backends.
# ---------------------------------------------------------------------------
FROM ${BUILD_BASE} AS build

# Extra cmake flags appended verbatim (e.g. -DPARAKEET_GGML_CUDA=ON).
ARG CMAKE_EXTRA_ARGS=""
# CUDA architectures, passed as a quoted CMAKE_CUDA_ARCHITECTURES list so the
# ';' separator survives the shell (e.g. "90;121-real"). Empty = let ggml pick
# its default broad list. Kept separate from CMAKE_EXTRA_ARGS for that reason.
ARG CUDA_ARCHS=""

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# CMake auto-applies the in-tree ggml patches during configure via
# scripts/apply_ggml_patches.sh, which uses `git apply` and therefore needs
# third_party/ggml to be a git repo. Re-init it as a throwaway repo so this
# works regardless of how the submodule arrived in the build context.
RUN rm -rf third_party/ggml/.git && git -C third_party/ggml init -q

# GGML_NATIVE=OFF keeps the binary portable across the CPUs that will pull the
# published image (no host-specific ISA extensions baked in). GGML_LLAMAFILE
# stays on (forced by CMakeLists) for the tinyBLAS SGEMM speedup.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DPARAKEET_BUILD_CLI=ON \
        -DPARAKEET_BUILD_SERVER=ON \
        -DPARAKEET_BUILD_TESTS=OFF \
        ${CMAKE_EXTRA_ARGS} \
        ${CUDA_ARCHS:+"-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"} \
    && cmake --build build -j"$(nproc)"

# Stage both binaries and every backend shared library (CPU, and CUDA when
# built) into a clean prefix the runtime stages copy from. The cli and server
# images each pick only the binary they ship.
RUN mkdir -p /install/bin /install/lib \
    && cp build/examples/cli/parakeet-cli /install/bin/ \
    && cp build/examples/server/parakeet-server /install/bin/ \
    && find build -name '*.so*' -exec cp -av {} /install/lib/ \;

# ---------------------------------------------------------------------------
# runtime-base: shared slim layer with the ggml backend libraries. The cli and
# server targets below add their own binary and entrypoint on top.
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /install/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /work

# ---------------------------------------------------------------------------
# runtime-server: the OpenAI-compatible HTTP server. Binds 0.0.0.0 so the
# published port is reachable from outside the container; curl is added so
# `--model <alias>` can fetch a published model on first run.
# ---------------------------------------------------------------------------
FROM runtime-base AS runtime-server
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /install/bin/parakeet-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["parakeet-server", "--host", "0.0.0.0"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# runtime: the cli image. Kept last so a plain `docker build .` (no --target)
# still produces the cli image exactly as before.
# ---------------------------------------------------------------------------
FROM runtime-base AS runtime
COPY --from=build /install/bin/parakeet-cli /usr/local/bin/
ENTRYPOINT ["parakeet-cli"]
CMD ["--help"]
