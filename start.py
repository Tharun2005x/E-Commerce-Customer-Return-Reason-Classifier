import os
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    port = os.getenv("PORT", "8501")
    api_port = os.getenv("API_PORT", "8000")

    env = os.environ.copy()
    env.setdefault("API_URL", f"http://127.0.0.1:{api_port}")

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            api_port,
        ],
        env=env,
    )

    health_url = f"http://127.0.0.1:{api_port}/health"
    for _ in range(30):
        if api_proc.poll() is not None:
            return api_proc.returncode or 1
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    ui_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            port,
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        env=env,
    )

    def shutdown(*_):
        for proc in (ui_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        while True:
            if ui_proc.poll() is not None:
                return ui_proc.returncode or 0
            if api_proc.poll() is not None:
                return api_proc.returncode or 1
            time.sleep(1)
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
