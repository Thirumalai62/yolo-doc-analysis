#!/usr/bin/env bash
set -euo pipefail

clone3_compat=$(printf '%s' "${EXECD_CLONE3_COMPAT:-}" | tr '[:upper:]' '[:lower:]')
case "${clone3_compat}" in
"" | 0 | false | off | no) ;;
1 | true | yes | on | reexec)
    if [[ -z "${_CODE_INTERPRETER_CLONE3_WRAPPED:-}" ]]; then
        export _CODE_INTERPRETER_CLONE3_WRAPPED=1
        exec /usr/local/bin/clone3-workaround /bin/bash "$0" "$@"
    fi
    unset EXECD_CLONE3_COMPAT
    ;;
*)
    echo "Invalid EXECD_CLONE3_COMPAT=${EXECD_CLONE3_COMPAT:-}." >&2
    exit 1
    ;;
esac

source /opt/code-interpreter/code-interpreter-env.sh python "${PYTHON_VERSION:-3.13}"
source /opt/code-interpreter/code-interpreter-env.sh node "${NODE_VERSION:-22}"

if [[ -n "${EXECD_ENVS:-}" ]]; then
    mkdir -p "$(dirname "${EXECD_ENVS}")" 2>/dev/null || true
    printf 'PATH=%s\n' "${PATH}" >>"${EXECD_ENVS}" 2>/dev/null || true
fi

export JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime
mkdir -p \
    /tmp/jupyter-runtime \
    /tmp/code-interpreter \
    /tmp/doc-detector/jobs \
    /tmp/doc-detector/cache/ultralytics \
    /tmp/doc-detector/cache/matplotlib

exec jupyter notebook \
    --ip=127.0.0.1 \
    --port="${JUPYTER_PORT:-44771}" \
    --allow-root \
    --no-browser \
    --IdentityProvider.token="${JUPYTER_TOKEN:-opensandboxcodeinterpreterjupyter}" \
    "$@" \
    >/tmp/code-interpreter/jupyter.log 2>&1
