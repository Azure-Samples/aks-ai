"""
Ray Worker Node Simulator - Tests performance with the head service.

Performs three test phases:
  1. DNS Resolution  - Resolve the head service hostname, measure latency
  2. Job Metadata    - Fetch job list, retrieve details, update status via HTTP
  3. Object Store    - Read and write large payloads, measure throughput
"""

import hashlib
import json
import os
import socket
import sys
import time
import urllib.request
import urllib.error

HEAD_SERVICE = os.environ.get("HEAD_SERVICE", "ray-head-svc")
HEAD_PORT = int(os.environ.get("HEAD_PORT", "8265"))
WORKER_ID = os.environ.get("HOSTNAME", "worker-unknown")
OBJECT_CHUNK_SIZE_MB = int(os.environ.get("OBJECT_CHUNK_SIZE_MB", "64"))
NUM_THROUGHPUT_ROUNDS = int(os.environ.get("NUM_THROUGHPUT_ROUNDS", "5"))
RETRY_INTERVAL = 5
MAX_RETRIES = 24  # 2 minutes


def log(msg):
    print(f"[WORKER {WORKER_ID}] {msg}", flush=True)


def phase_dns_resolution():
    """Phase 1: Resolve head service DNS and measure latency."""
    log("=" * 60)
    log("PHASE 1: DNS Resolution")
    log("=" * 60)

    fqdn_variants = [
        HEAD_SERVICE,
        f"{HEAD_SERVICE}.default.svc.cluster.local",
    ]

    for name in fqdn_variants:
        try:
            start = time.monotonic()
            results = socket.getaddrinfo(name, HEAD_PORT, socket.AF_INET, socket.SOCK_STREAM)
            elapsed_ms = (time.monotonic() - start) * 1000
            ips = list({r[4][0] for r in results})
            log(f"  Resolved {name} -> {ips} in {elapsed_ms:.2f} ms")
        except socket.gaierror as e:
            log(f"  FAILED to resolve {name}: {e}")

    # Measure repeated resolution latency
    latencies = []
    for _ in range(10):
        start = time.monotonic()
        socket.getaddrinfo(HEAD_SERVICE, HEAD_PORT, socket.AF_INET, socket.SOCK_STREAM)
        latencies.append((time.monotonic() - start) * 1000)

    avg = sum(latencies) / len(latencies)
    log(f"  DNS avg latency (10 lookups): {avg:.2f} ms  "
        f"min={min(latencies):.2f} ms  max={max(latencies):.2f} ms")
    return True


def http_get(path):
    url = f"http://{HEAD_SERVICE}:{HEAD_PORT}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def http_post(path, data):
    url = f"http://{HEAD_SERVICE}:{HEAD_PORT}{path}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def wait_for_head():
    """Wait until the head service is reachable."""
    log(f"Waiting for head service {HEAD_SERVICE}:{HEAD_PORT} ...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            status, body = http_get("/healthz")
            if status == 200:
                log(f"  Head service is ready (attempt {attempt})")
                return True
        except Exception:
            pass
        log(f"  Attempt {attempt}/{MAX_RETRIES} - retrying in {RETRY_INTERVAL}s")
        time.sleep(RETRY_INTERVAL)
    log("  FAILED: head service not reachable")
    return False


def phase_job_metadata():
    """Phase 2: Fetch and update job metadata via HTTP."""
    log("=" * 60)
    log("PHASE 2: Job Metadata & Status (HTTP)")
    log("=" * 60)

    # List all jobs
    start = time.monotonic()
    status, data = http_get("/api/jobs")
    elapsed_ms = (time.monotonic() - start) * 1000
    job_ids = [j["job_id"] for j in data["jobs"]]
    log(f"  Listed {len(job_ids)} jobs in {elapsed_ms:.1f} ms: {job_ids}")

    # Get details for each job
    for job_id in job_ids:
        start = time.monotonic()
        status, job = http_get(f"/api/jobs/{job_id}")
        elapsed_ms = (time.monotonic() - start) * 1000
        log(f"  Job {job_id}: status={job['status']}  "
            f"name={job['name']}  latency={elapsed_ms:.1f} ms")

    # Update status for first job
    target_job = job_ids[0]
    start = time.monotonic()
    status, result = http_post(f"/api/jobs/{target_job}/status", {
        "status": "RUNNING",
        "worker_id": WORKER_ID,
    })
    elapsed_ms = (time.monotonic() - start) * 1000
    log(f"  Updated {target_job} -> RUNNING  latency={elapsed_ms:.1f} ms")

    return True


def phase_object_store_throughput():
    """Phase 3: Measure read/write throughput with the head's object store."""
    log("=" * 60)
    log("PHASE 3: Object Store Throughput")
    log("=" * 60)

    chunk_bytes = OBJECT_CHUNK_SIZE_MB * 1024 * 1024

    # --- READ throughput ---
    read_rates = []
    for i in range(1, NUM_THROUGHPUT_ROUNDS + 1):
        url = f"http://{HEAD_SERVICE}:{HEAD_PORT}/api/object_store/read"
        req = urllib.request.Request(url)
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            expected_checksum = resp.headers.get("X-Checksum")
            data = resp.read()
        elapsed = time.monotonic() - start
        actual_checksum = hashlib.sha256(data).hexdigest()
        mb = len(data) / (1024 * 1024)
        rate = mb / elapsed
        read_rates.append(rate)
        integrity = "OK" if actual_checksum == expected_checksum else "MISMATCH"
        log(f"  READ  round {i}/{NUM_THROUGHPUT_ROUNDS}: "
            f"{mb:.0f} MB in {elapsed:.3f}s = {rate:.1f} MB/s  "
            f"checksum={integrity}")

    avg_read = sum(read_rates) / len(read_rates)
    log(f"  READ  avg={avg_read:.1f} MB/s  "
        f"min={min(read_rates):.1f}  max={max(read_rates):.1f}")

    # --- WRITE throughput ---
    write_rates = []
    payload = os.urandom(chunk_bytes)
    payload_checksum = hashlib.sha256(payload).hexdigest()
    for i in range(1, NUM_THROUGHPUT_ROUNDS + 1):
        url = f"http://{HEAD_SERVICE}:{HEAD_PORT}/api/object_store/write"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        elapsed = time.monotonic() - start
        mb = chunk_bytes / (1024 * 1024)
        rate = mb / elapsed
        write_rates.append(rate)
        integrity = "OK" if result["checksum"] == payload_checksum else "MISMATCH"
        log(f"  WRITE round {i}/{NUM_THROUGHPUT_ROUNDS}: "
            f"{mb:.0f} MB in {elapsed:.3f}s = {rate:.1f} MB/s  "
            f"checksum={integrity}")

    avg_write = sum(write_rates) / len(write_rates)
    log(f"  WRITE avg={avg_write:.1f} MB/s  "
        f"min={min(write_rates):.1f}  max={max(write_rates):.1f}")

    return avg_read, avg_write


def main():
    log("Starting Ray worker benchmark")
    log(f"  HEAD_SERVICE={HEAD_SERVICE}  HEAD_PORT={HEAD_PORT}")
    log(f"  OBJECT_CHUNK_SIZE_MB={OBJECT_CHUNK_SIZE_MB}")
    log(f"  NUM_THROUGHPUT_ROUNDS={NUM_THROUGHPUT_ROUNDS}")

    if not wait_for_head():
        sys.exit(1)

    phase_dns_resolution()
    phase_job_metadata()
    avg_read, avg_write = phase_object_store_throughput()

    # Final summary
    log("=" * 60)
    log("TEST COMPLETE - SUMMARY")
    log("=" * 60)
    log(f"  DNS resolution:     PASS")
    log(f"  Job metadata HTTP:  PASS")
    log(f"  Object store READ:  {avg_read:.1f} MB/s avg")
    log(f"  Object store WRITE: {avg_write:.1f} MB/s avg")

    # Mark job as complete
    http_post(f"/api/jobs/job-001/status", {
        "status": "SUCCEEDED",
        "worker_id": WORKER_ID,
    })
    log("Done.")


if __name__ == "__main__":
    main()
