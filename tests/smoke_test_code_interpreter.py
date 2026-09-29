"""Exercise focused Python and Node.js kernels through Jupyter's API."""

from __future__ import annotations

import json
import time
import urllib.request
import uuid

import websocket


BASE_URL = "http://127.0.0.1:44771"
WS_URL = "ws://127.0.0.1:44771"
TOKEN = "opensandboxcodeinterpreterjupyter"


def wait_for_kernels(timeout: int = 60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"{BASE_URL}/api/kernelspecs?token={TOKEN}", timeout=3
            ) as response:
                specs = json.loads(response.read()).get("kernelspecs", {})
            if {"python", "jslab", "tslab"}.issubset(specs):
                return specs
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Python and Node.js kernels did not become ready.")


class Kernel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.kernel_id = ""
        self.socket = None

    def __enter__(self):
        request = urllib.request.Request(
            f"{BASE_URL}/api/kernels?token={TOKEN}",
            data=json.dumps({"name": self.name}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            self.kernel_id = json.loads(response.read())["id"]
        self.socket = websocket.create_connection(
            f"{WS_URL}/api/kernels/{self.kernel_id}/channels?token={TOKEN}",
            timeout=30,
        )
        return self

    def execute(self, code: str, timeout: int = 60) -> str:
        socket = self.socket
        if socket is None:
            raise RuntimeError("Kernel WebSocket is not connected.")
        message_id = str(uuid.uuid4())
        message = {
            "header": {
                "msg_id": message_id,
                "username": "smoke-test",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
            "channel": "shell",
        }
        socket.send(json.dumps(message))
        output: list[str] = []
        status = None
        idle = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload = json.loads(socket.recv())
            if payload.get("parent_header", {}).get("msg_id") != message_id:
                continue
            message_type = payload.get("msg_type")
            content = payload.get("content", {})
            if message_type == "stream":
                output.append(content.get("text", ""))
            elif message_type == "execute_result":
                output.append(content.get("data", {}).get("text/plain", ""))
            elif message_type == "error":
                raise RuntimeError(f"{content.get('ename')}: {content.get('evalue')}")
            elif message_type == "execute_reply":
                status = content.get("status")
            elif message_type == "status" and content.get("execution_state") == "idle":
                idle = True
            if idle and status is not None:
                if status != "ok":
                    raise RuntimeError(f"{self.name} execution status: {status}")
                return "".join(output).strip()
        raise TimeoutError(f"{self.name} execution did not complete.")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.socket is not None:
            self.socket.close()
        if self.kernel_id:
            request = urllib.request.Request(
                f"{BASE_URL}/api/kernels/{self.kernel_id}?token={TOKEN}",
                method="DELETE",
            )
            try:
                urllib.request.urlopen(request, timeout=5).close()
            except Exception:
                pass


def main() -> None:
    specs = wait_for_kernels()
    print(f"Kernels: {sorted(specs)}")

    with Kernel("python") as kernel:
        assert "PYTHON_42" in kernel.execute('value = 42\nprint(f"PYTHON_{value}")')
        assert "PYTHON_REUSED_52" in kernel.execute(
            'print(f"PYTHON_REUSED_{value + 10}")'
        )

    with Kernel("bash") as kernel:
        assert "BASH_66" in kernel.execute('printf "BASH_%s\\n" "$((60 + 6))"')

    with Kernel("jslab") as kernel:
        assert "JAVASCRIPT_55" in kernel.execute(
            'console.log("JAVASCRIPT_" + (50 + 5));'
        )

    with Kernel("tslab") as kernel:
        assert "TYPESCRIPT_77" in kernel.execute(
            'const value: number = 70 + 7; console.log("TYPESCRIPT_" + value);'
        )

    print("Python state reuse and Bash/JavaScript/TypeScript kernels passed.")


if __name__ == "__main__":
    main()
