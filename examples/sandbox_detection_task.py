"""Reference OpenSandbox flow for one prewarmed URL-detection task.

Install `opensandbox` and `opensandbox-code-interpreter` in the platform process,
not inside the detector image. Configure the required environment variables
before running this example.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import os
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit

from code_interpreter import CodeInterpreter, SupportedLanguage  # type: ignore[import-not-found]
from opensandbox import Sandbox  # type: ignore[import-not-found]
from opensandbox.config import ConnectionConfig  # type: ignore[import-not-found]
from opensandbox.models.filesystem import WriteEntry  # type: ignore[import-not-found]
from opensandbox.models.sandboxes import (  # type: ignore[import-not-found]
    NetworkPolicy,
    NetworkRule,
)


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def required_environment(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def run_task(sandbox, interpreter, context, task: dict[str, object]) -> dict:
    job_id = str(task["job_id"])
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Task job_id contains unsupported characters.")
    task_file = f"/tmp/doc-detector/task-{job_id}.json"
    await sandbox.files.write_files(
        [WriteEntry(path=task_file, data=json.dumps(task), mode=600)]
    )
    try:
        execution = await interpreter.codes.run(
            "import json\n"
            "from pathlib import Path\n"
            f"task_path = Path({task_file!r})\n"
            "try:\n"
            "    task = json.loads(task_path.read_text(encoding='utf-8'))\n"
            "finally:\n"
            "    task_path.unlink(missing_ok=True)\n"
            "detection_result = detector.detect(**task)\n"
            "{'result': detection_result.as_dict(), 'model_identity': detector.model_identity}",
            context=context,
        )
    finally:
        try:
            await sandbox.files.delete_files([task_file])
        except Exception:
            pass
    if execution.error:
        raise RuntimeError(f"Detection failed: {execution.error.value}")

    output_dir = str(task["output_dir"])
    result_path = str(PurePosixPath(output_dir) / "result.json")
    result = json.loads(await sandbox.files.read_file(result_path))
    print(json.dumps(result, indent=2))

    # Binary artifacts can be read or streamed here before later upload logic.
    crops = result["artifacts"]["crops"]
    if crops:
        first_crop = str(PurePosixPath(output_dir) / crops[0])
        first_crop_bytes = await sandbox.files.read_bytes(first_crop)
        print(f"First crop is available: {first_crop} ({len(first_crop_bytes)} bytes)")
    return result


async def main() -> None:
    job_id = os.getenv("DETECTOR_JOB_ID", "example-job")
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("DETECTOR_JOB_ID contains unsupported characters.")

    pdf_url = required_environment("DETECTOR_PDF_URL")
    pdf_host = urlsplit(pdf_url).hostname
    if not pdf_host:
        raise ValueError("DETECTOR_PDF_URL must include a hostname.")
    allowed_hosts = {pdf_host}
    allowed_hosts.update(
        host.strip()
        for host in os.getenv("DETECTOR_EGRESS_HOSTS", "").split(",")
        if host.strip()
    )
    tasks = [{
        "job_id": job_id,
        "output_dir": f"/tmp/doc-detector/jobs/{job_id}",
        "pdf_url": pdf_url,
        "outputs": ["json", "crops", "annotated_pages", "rendered_pages"],
    }]
    second_pdf_url = os.getenv("DETECTOR_SECOND_PDF_URL")
    if second_pdf_url:
        second_job_id = os.getenv("DETECTOR_SECOND_JOB_ID", "example-job-2")
        if not JOB_ID_PATTERN.fullmatch(second_job_id):
            raise ValueError("DETECTOR_SECOND_JOB_ID contains unsupported characters.")
        second_host = urlsplit(second_pdf_url).hostname
        if not second_host:
            raise ValueError("DETECTOR_SECOND_PDF_URL must include a hostname.")
        allowed_hosts.add(second_host)
        tasks.append({
            "job_id": second_job_id,
            "output_dir": f"/tmp/doc-detector/jobs/{second_job_id}",
            "pdf_url": second_pdf_url,
            "outputs": ["json", "crops", "annotated_pages", "rendered_pages"],
        })
    config = ConnectionConfig(
        domain=os.getenv("SANDBOX_DOMAIN", "localhost:8080"),
        api_key=os.getenv("SANDBOX_API_KEY"),
        request_timeout=timedelta(minutes=2),
    )
    sandbox = await Sandbox.create(
        required_environment("SANDBOX_IMAGE"),
        connection_config=config,
        entrypoint=["/opt/code-interpreter/code-interpreter.sh"],
        env={"PYTHON_VERSION": os.getenv("SANDBOX_PYTHON_VERSION", "3.13")},
        resource={
            "cpu": os.getenv("SANDBOX_CPU", "4"),
            "memory": os.getenv("SANDBOX_MEMORY", "8Gi"),
        },
        network_policy=NetworkPolicy(
            defaultAction="deny",
            egress=[NetworkRule(action="allow", target=host) for host in sorted(allowed_hosts)],
        ),
        timeout=timedelta(minutes=int(os.getenv("SANDBOX_TTL_MINUTES", "60"))),
        ready_timeout=timedelta(minutes=5),
    )

    try:
        interpreter = await CodeInterpreter.create(sandbox=sandbox)
        context = await interpreter.codes.create_context(SupportedLanguage.PYTHON)

        warmup = await interpreter.codes.run(
            "from doc_detector import get_detector\n"
            "detector = get_detector(model_version='r7')\n"
            "detector.warmup()",
            context=context,
        )
        if warmup.error:
            raise RuntimeError(f"Detector warmup failed: {warmup.error.value}")

        # Every task reuses the detector variable in this same Python context.
        for task in tasks:
            await run_task(sandbox, interpreter, context, task)
    finally:
        await sandbox.destroy()


if __name__ == "__main__":
    asyncio.run(main())
