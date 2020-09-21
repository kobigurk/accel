#!/bin/bash
set -xue

NIGHTLY=nightly-2020-09-20
rustup toolchain add ${NIGHTLY}
rustup target add nvptx64-nvidia-cuda --toolchain ${NIGHTLY}
cargo install ptx-linker -f
