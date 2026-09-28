#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_SOURCE=${MODEL_SOURCE:-"${ROOT}/runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt"}
EXPECTED_MODEL_SHA256=2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76
IMAGE=${IMAGE:-legal-notice-detector:r7}
BASE_IMAGE=${BASE_IMAGE:-opensandbox/code-interpreter:v1.1.0}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}
RELEASE=${RELEASE:-false}
STAGING_DIR="${ROOT}/.build/opensandbox"

if [[ "${RELEASE}" == "true" && "${BASE_IMAGE}" != *@sha256:* ]]; then
    echo "Release builds require BASE_IMAGE to include an immutable sha256 digest." >&2
    exit 1
fi

if [[ ! -f "${MODEL_SOURCE}" ]]; then
    echo "R7 model does not exist: ${MODEL_SOURCE}" >&2
    exit 1
fi

actual_sha256=$(sha256sum "${MODEL_SOURCE}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
    echo "R7 model checksum mismatch." >&2
    echo "Expected: ${EXPECTED_MODEL_SHA256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    exit 1
fi

mkdir -p "${STAGING_DIR}"
trap 'rm -rf "${STAGING_DIR}"' EXIT
cp "${MODEL_SOURCE}" "${STAGING_DIR}/legal_notice_r7.pt"

docker build \
    --platform linux/amd64 \
    --file "${ROOT}/Dockerfile.opensandbox" \
    --tag "${IMAGE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    "${ROOT}"

echo "Built ${IMAGE}"
