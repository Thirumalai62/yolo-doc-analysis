#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_VERSION=${MODEL_VERSION:-r7}
MANIFEST_PATH="${ROOT}/model-manifest.json"
IMAGE=${IMAGE:-legal-notice-detector:r7}
VARIANT=${VARIANT:-slim}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}
RELEASE=${RELEASE:-false}
STAGING_DIR="${ROOT}/.build/opensandbox"

case "${VARIANT}" in
    slim)
        DOCKERFILE="${ROOT}/Dockerfile.opensandbox"
        BASE_IMAGE=${BASE_IMAGE:-python:3.13.13-slim-bookworm@sha256:355bfa66770995d7e9a0da4b3473b44d0cb451f6b56f5615ad9c39e3c4eca03f}
        ;;
    full)
        DOCKERFILE="${ROOT}/Dockerfile.opensandbox.full"
        BASE_IMAGE=${BASE_IMAGE:-opensandbox/code-interpreter:v1.1.0}
        ;;
    *)
        echo "VARIANT must be 'slim' or 'full'." >&2
        exit 1
        ;;
esac

model_output=$(python3 - "${MANIFEST_PATH}" "${MODEL_VERSION}" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
model_version = sys.argv[2]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
try:
    model = manifest["models"][model_version]
except KeyError as error:
    raise SystemExit(f"Model version {model_version!r} is not defined in {manifest_path}.") from error
print(model["file"])
print(model["sha256"])
PY
)
readarray -t model_values <<<"${model_output}"
MODEL_FILE=${model_values[0]}
EXPECTED_MODEL_SHA256=${model_values[1]}
PACKAGED_MODEL="${ROOT}/${MODEL_FILE}"
if [[ -z "${MODEL_SOURCE:-}" ]]; then
    MODEL_SOURCE="${PACKAGED_MODEL}"
fi

if [[ "${RELEASE}" == "true" && "${BASE_IMAGE}" != *@sha256:* ]]; then
    echo "Release builds require BASE_IMAGE to include an immutable sha256 digest." >&2
    exit 1
fi

if [[ ! -f "${MODEL_SOURCE}" ]]; then
    echo "Model '${MODEL_VERSION}' does not exist: ${MODEL_SOURCE}." >&2
    echo "Run 'git lfs pull' in a deployment clone or set MODEL_SOURCE." >&2
    exit 1
fi
if head -c 128 "${MODEL_SOURCE}" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
    echo "Model '${MODEL_VERSION}' is an unresolved Git LFS pointer. Run 'git lfs pull'." >&2
    exit 1
fi

actual_sha256=$(sha256sum "${MODEL_SOURCE}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
    echo "Model '${MODEL_VERSION}' checksum mismatch." >&2
    echo "Expected: ${EXPECTED_MODEL_SHA256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    exit 1
fi

mkdir -p "${STAGING_DIR}"
trap 'rm -rf "${STAGING_DIR}"' EXIT
cp "${MODEL_SOURCE}" "${STAGING_DIR}/legal_notice_r7.pt"

docker build \
    --platform linux/amd64 \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    "${ROOT}"

echo "Built ${IMAGE} (${VARIANT} runtime)"
