"""
Ray Head Node Simulator - HTTP server for job metadata and iperf3 server.

Exposes endpoints that mimic Ray head service:
  - GET  /api/jobs              List all jobs
  - GET  /api/jobs/<id>         Get job metadata and status
  - POST /api/jobs/<id>/status  Update job status from worker

Also runs an iperf3 server on IPERF_PORT (default 5201) for network
throughput and RTT measurement by worker nodes.
"""

import http.server
import json
import os
import subprocess
import time
import threading

HOST = "0.0.0.0"
PORT = int(os.environ.get("HEAD_PORT", "8265"))
IPERF_PORT = int(os.environ.get("IPERF_PORT", "5201"))

# In-memory job registry
jobs = {
    "job-001": {
        "job_id": "job-001",
        "name": "distributed-training",
        "status": "PENDING",
        "submitted_at": time.time(),
        "metadata": {
            "num_workers": 1,
            "gpus_per_worker": 1,
            "framework": "pytorch",
            "entrypoint": "python train.py",
        },
    },
    "job-002": {
        "job_id": "job-002",
        "name": "data-preprocessing",
        "status": "PENDING",
        "submitted_at": time.time(),
        "metadata": {
            "num_workers": 4,
            "gpus_per_worker": 0,
            "framework": "ray-data",
            "entrypoint": "python preprocess.py",
        },
    },
}
jobs_lock = threading.Lock()


class HeadHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[HEAD] {self.client_address[0]} - {fmt % args}")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/jobs":
            with jobs_lock:
                self._send_json({"jobs": list(jobs.values())})

        elif self.path.startswith("/api/jobs/") and "/status" not in self.path:
            job_id = self.path.split("/")[-1]
            with jobs_lock:
                job = jobs.get(job_id)
            if job:
                self._send_json(job)
            else:
                self._send_json({"error": "job not found"}, 404)

        elif self.path == "/healthz":
            self._send_json({"status": "ok"})

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))

        if self.path.startswith("/api/jobs/") and self.path.endswith("/status"):
            job_id = self.path.split("/")[-2]
            body = json.loads(self.rfile.read(content_len))
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = body.get("status", jobs[job_id]["status"])
                    if "worker_id" in body:
                        jobs[job_id].setdefault("workers", []).append(body["worker_id"])
                    self._send_json(jobs[job_id])
                else:
                    self._send_json({"error": "job not found"}, 404)

        else:
            self._send_json({"error": "not found"}, 404)


def start_iperf_server():
    """Start iperf3 server in the background."""
    print(f"[HEAD] Starting iperf3 server on port {IPERF_PORT}")
    proc = subprocess.Popen(
        ["iperf3", "--server", "--port", str(IPERF_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        print(f"[HEAD-IPERF] {line.decode().rstrip()}")
    rc = proc.wait()
    print(f"[HEAD] iperf3 server exited with rc={rc}")


if __name__ == "__main__":
    # Start iperf3 server in a daemon thread (restarts after each client)
    iperf_thread = threading.Thread(target=start_iperf_server, daemon=True)
    iperf_thread.start()

    server = http.server.ThreadingHTTPServer((HOST, PORT), HeadHandler)
    print(f"[HEAD] Ray head simulator listening on {HOST}:{PORT}")
    print(f"[HEAD] iperf3 server listening on {IPERF_PORT}")
    server.serve_forever()
