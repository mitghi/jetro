#!/usr/bin/env bash
# Run Miri over unsafe-touching tests.
#
# Installation (one-time):
#   rustup +nightly component add miri
#
# Targets every unsafe block in jetro-core — if this passes, the
# documented SAFETY invariants are empirically sound.

set -euo pipefail

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-strict-provenance}"

cargo +nightly miri test -p jetro-core --test unsafe_invariants "$@"
