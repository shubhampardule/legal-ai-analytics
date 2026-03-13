from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "integration" / "full_stack_smoke_test.json"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8001
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5174


def wait_for_url(url: str, timeout_seconds: float = 180.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    with httpx.Client(follow_redirects=True, timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(url)
                if response.status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}. Last error: {last_error}")


def terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait(timeout=20)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    backend_log = REPORT_PATH.with_name("backend_smoke.log")
    frontend_log = REPORT_PATH.with_name("frontend_smoke.log")

    backend_process = None
    frontend_process = None

    try:
        with backend_log.open("w", encoding="utf-8") as backend_handle:
            backend_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.app.main:app",
                    "--host",
                    BACKEND_HOST,
                    "--port",
                    str(BACKEND_PORT),
                ],
                cwd=PROJECT_ROOT,
                stdout=backend_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        npm_command = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm_command:
            raise RuntimeError("npm executable not found for frontend smoke test.")

        frontend_env = os.environ.copy()
        frontend_env["VITE_API_BASE_URL"] = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
        with frontend_log.open("w", encoding="utf-8") as frontend_handle:
            frontend_process = subprocess.Popen(
                [
                    npm_command,
                    "run",
                    "dev",
                    "--",
                    "--host",
                    FRONTEND_HOST,
                    "--port",
                    str(FRONTEND_PORT),
                    "--strictPort",
                ],
                cwd=FRONTEND_DIR,
                env=frontend_env,
                stdout=frontend_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )

        backend_origin = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
        frontend_origin = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"

        wait_for_url(f"{backend_origin}/health")
        wait_for_url(frontend_origin)

        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            frontend_root = client.get(frontend_origin)
            cors_preflight = client.options(
                f"{backend_origin}/api/v1/analyze",
                headers={
                    "Origin": frontend_origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            analyze_response = client.post(
                f"{backend_origin}/api/v1/analyze",
                headers={
                    "Origin": frontend_origin,
                    "Content-Type": "application/json",
                },
                json={
                    "case_id": "1980_211",
                    "top_k": 3,
                    "include_explanation": True,
                    "include_similar_cases": True,
                },
            )

        analyze_body = analyze_response.json()
        report = {
            "frontend_origin": frontend_origin,
            "backend_origin": backend_origin,
            "checks": {
                "frontend_root_status": frontend_root.status_code,
                "frontend_root_contains_root_div": '<div id="root"></div>' in frontend_root.text,
                "cors_preflight_status": cors_preflight.status_code,
                "cors_allow_origin": cors_preflight.headers.get("access-control-allow-origin"),
                "analyze_status": analyze_response.status_code,
                "analyze_envelope_status": analyze_body.get("status"),
                "predicted_label": analyze_body["data"]["prediction"]["predicted_label"],
                "retrieval_result_count": len(analyze_body["data"]["retrieval"]["results"]),
                "first_retrieved_case_id": analyze_body["data"]["retrieval"]["results"][0]["case_id"],
            },
            "artifacts": {
                "backend_log": str(backend_log),
                "frontend_log": str(frontend_log),
            },
        }

        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        terminate_process(frontend_process)
        terminate_process(backend_process)


if __name__ == "__main__":
    main()
