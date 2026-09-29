#!/usr/bin/env bash

# Focused version selector compatible with the OpenSandbox entrypoint contract.
language=${1:-}
version=${2:-}

append_env() {
    if [[ -n "${EXECD_ENVS:-}" ]]; then
        mkdir -p "$(dirname "${EXECD_ENVS}")" 2>/dev/null || true
        printf '%s=%s\n' "$1" "$2" >>"${EXECD_ENVS}" 2>/dev/null || true
    fi
}

case "${language}" in
python)
    version=${version:-3.13}
    if [[ "${version}" != "3.13" ]]; then
        echo "Python ${version} is not installed; supported version: 3.13." >&2
        return 1
    fi
    export PATH="/usr/local/bin:/opt/node/v22.2.0/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
    append_env PATH "${PATH}"
    echo "Switched to Python $(python --version)"
    ;;
node)
    version=${version:-22}
    if [[ "${version}" != "22" && "${version}" != "22.2.0" ]]; then
        echo "Node.js ${version} is not installed; supported version: 22." >&2
        return 1
    fi
    export PATH="/opt/node/v22.2.0/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
    append_env PATH "${PATH}"
    echo "Switched to Node $(node --version)"
    ;;
*)
    echo "Unsupported language '${language}'. Supported languages: python, node." >&2
    return 1
    ;;
esac
