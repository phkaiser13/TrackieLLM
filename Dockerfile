# Dockerfile for TrackieLLM Production Runtime
# This is a multi-stage build to keep the final image small.

# --- Builder Stage ---
# This stage builds the application and all its dependencies.
FROM ubuntu:22.04 AS builder

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    # Add other project-specific dependencies here (e.g., for audio, vision)
    # libssl-dev, etc.
  && rm -rf /var/lib/apt/lists/*

# Install Rust
# Using rustup is preferred to get a recent version of the compiler.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the source code into the container
WORKDIR /app
COPY . .

# Configure and build the application
# The exact CMake flags will depend on the target.
# For a generic build, we might not enable GPU support.
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --parallel

# --- Final Stage ---
# This stage creates the final, lean runtime image.
FROM ubuntu:22.04

# Install only runtime dependencies, not build tools.
RUN apt-get update && apt-get install -y \
    # Add only necessary runtime libs, e.g., libssl, etc.
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy the compiled application from the builder stage
COPY --from=builder /app/build/trackie_executable /usr/local/bin/trackie_executable

# Set the entrypoint for the container
# This assumes the main application is a single executable.
ENTRYPOINT ["/usr/local/bin/trackie_executable"]
CMD []
