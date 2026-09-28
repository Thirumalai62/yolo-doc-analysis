#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-legal-notice-detector:r7}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}

docker run --rm \
    --platform linux/amd64 \
    --entrypoint /bin/bash \
    --env "PYTHON_VERSION=${PYTHON_VERSION}" \
    "${IMAGE}" \
    -lc 'source /opt/code-interpreter/code-interpreter-env.sh python "${PYTHON_VERSION}" && python -c "from doc_detector import get_detector; detector = get_detector(); print(detector.warmup())"'

echo "OpenSandbox detector image smoke test passed: ${IMAGE}"
