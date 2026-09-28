#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PDF_URL=${1:-}
IMAGE=${IMAGE:-legal-notice-detector:r7}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${ROOT}/output"}
DETECTOR_JOB_ID=${DETECTOR_JOB_ID:-}
DETECTOR_OUTPUTS=${DETECTOR_OUTPUTS:-json,crops,annotated_pages,rendered_pages}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}
DEV=${DEV:-false}

if [[ -z "${PDF_URL}" ]]; then
    echo "Usage: scripts/run-local-detector.sh <direct-pdf-url>" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
docker image inspect "${IMAGE}" >/dev/null

docker_args=(
    run --rm -i
    --platform linux/amd64
    --entrypoint /bin/bash
    --env DETECTOR_PDF_URL
    --env DETECTOR_JOB_ID
    --env DETECTOR_OUTPUTS
    --env "PYTHON_VERSION=${PYTHON_VERSION}"
    --mount "type=bind,source=${OUTPUT_ROOT},target=/tmp/doc-detector/jobs"
)
if [[ "${DEV}" == "true" ]]; then
    docker_args+=(
        --mount "type=bind,source=${ROOT}/doc_detector,target=/opt/doc-detector/doc_detector,readonly"
    )
fi
docker_args+=(
    "${IMAGE}"
    -lc 'source /opt/code-interpreter/code-interpreter-env.sh python "${PYTHON_VERSION}" && python -'
)

export DETECTOR_PDF_URL="${PDF_URL}"
export DETECTOR_JOB_ID
export DETECTOR_OUTPUTS
docker "${docker_args[@]}" < "${ROOT}/examples/local_detection_task.py"
echo "Detection outputs are under ${OUTPUT_ROOT}"
