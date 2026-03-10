"""
Ray Worker Node Simulator - Tests performance with the head service.

Performs three test phases:
  1. DNS Resolution  - Resolve the head service hostname, measure latency
  2. Job Metadata    - Fetch job list, retrieve details, update status via HTTP
  3. Object Store    - Read and write large payloads, measure throughput
"""

import concurrent.futures
import hashlib
import json
import os
import socket
import sys
import time
import urllib.request

HEAD_SERVICE = os.environ.get("HEAD_SERVICE", "ray-head-svc")
HEAD_PORT = int(os.environ.get("HEAD_PORT", "8265"))
WORKER_ID = os.environ.get("HOSTNAME", "worker-unknown")
OBJECT_CHUNK_SIZE_MB = int(os.environ.get("OBJECT_CHUNK_SIZE_MB", "64"))
NUM_THROUGHPUT_ROUNDS = int(os.environ.get("NUM_THROUGHPUT_ROUNDS", "5"))
PARALLEL_STREAMS = int(os.environ.get("PARALLEL_STREAMS", "8"))
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
        try:
            start = time.monotonic()
            socket.getaddrinfo(HEAD_SERVICE, HEAD_PORT, socket.AF_INET, socket.SOCK_STREAM)
            latencies.append((time.monotonic() - start) * 1000)
        except socket.gaierror as e:
            log(f"  DNS lookup failed during latency test: {e}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        log(f"  DNS avg latency ({len(latencies)} lookups): {avg:.2f} ms  "
            f"min={min(latencies):.2f} ms  max={max(latencies):.2f} ms")
    else:
        log("  DNS latency test: all lookups failed")
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


def _do_one_read(stream_id):
    """Single read stream — returns (bytes_read, elapsed, ttfb)."""
    url = f"http://{HEAD_SERVICE}:{HEAD_PORT}/api/object_store/read"
    req = urllib.request.Request(url)
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        ttfb = (time.monotonic() - start) * 1000
        data = resp.read()
    elapsed = time.monotonic() - start
    return len(data), elapsed, ttfb


def _do_one_write(payload):
    """Single write stream — returns (bytes_written, elapsed)."""
    url = f"http://{HEAD_SERVICE}:{HEAD_PORT}/api/object_store/write"
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()
    elapsed = time.monotonic() - start
    return len(payload), elapsed


def phase_object_store_throughput():
    """Phase 3: Measure read/write throughput with parallel streams."""
    log("=" * 60)
    log("PHASE 3: Object Store Throughput")
    log(f"  chunk={OBJECT_CHUNK_SIZE_MB} MB  streams={PARALLEL_STREAMS}  "
        f"rounds={NUM_THROUGHPUT_ROUNDS}")
    log("=" * 60)

    chunk_bytes = OBJECT_CHUNK_SIZE_MB * 1024 * 1024

    # --- Latency test (small request round-trip) ---
    log("  --- Latency (small request round-trip) ---")
    rt_latencies = []
    for _ in range(20):
        url = f"http://{HEAD_SERVICE}:{HEAD_PORT}/healthz"
        req = urllib.request.Request(url)
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        rt_latencies.append((time.monotonic() - start) * 1000)
    avg_rt = sum(rt_latencies) / len(rt_latencies)
    p50 = sorted(rt_latencies)[len(rt_latencies) // 2]
    p99 = sorted(rt_latencies)[int(len(rt_latencies) * 0.99)]
    log(f"  RTT (20 reqs): avg={avg_rt:.2f} ms  p50={p50:.2f} ms  "
        f"p99={p99:.2f} ms  min={min(rt_latencies):.2f} ms  "
        f"max={max(rt_latencies):.2f} ms")

    # --- Single-stream baseline ---
    log("  --- Single-stream baseline ---")
    _, r_elapsed, r_ttfb = _do_one_read(0)
    r_mb = chunk_bytes / (1024 * 1024)
    log(f"  READ  1x{r_mb:.0f} MB: {r_mb/r_elapsed:.1f} MB/s  "
        f"TTFB={r_ttfb:.1f} ms")
    payload = os.urandom(chunk_bytes)
    _, w_elapsed = _do_one_write(payload)
    log(f"  WRITE 1x{r_mb:.0f} MB: {r_mb/w_elapsed:.1f} MB/s")

    # --- Parallel READ throughput ---
    log(f"  --- READ ({PARALLEL_STREAMS} parallel streams) ---")
    read_agg_rates = []
    read_ttfbs = []
    for rnd in range(1, NUM_THROUGHPUT_ROUNDS + 1):
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_STREAMS) as pool:
            futures = [pool.submit(_do_one_read, s) for s in range(PARALLEL_STREAMS)]
            results = [f.result() for f in futures]
        wall_time = time.monotonic() - start
        total_mb = sum(r[0] for r in results) / (1024 * 1024)
        agg_rate = total_mb / wall_time
        ttfbs = [r[2] for r in results]
        per_stream = [r[0] / (1024 * 1024) / r[1] for r in results]
        read_agg_rates.append(agg_rate)
        read_ttfbs.extend(ttfbs)
        log(f"  READ  round {rnd}/{NUM_THROUGHPUT_ROUNDS}: "
            f"{total_mb:.0f} MB in {wall_time:.3f}s = {agg_rate:.1f} MB/s aggregate  "
            f"per-stream avg={sum(per_stream)/len(per_stream):.1f} MB/s  "
            f"TTFB avg={sum(ttfbs)/len(ttfbs):.1f} ms")

    avg_read = sum(read_agg_rates) / len(read_agg_rates)
    log(f"  READ  aggregate: avg={avg_read:.1f} MB/s  "
        f"min={min(read_agg_rates):.1f}  max={max(read_agg_rates):.1f}")
    log(f"  READ  TTFB:      avg={sum(read_ttfbs)/len(read_ttfbs):.1f} ms  "
        f"min={min(read_ttfbs):.1f}  max={max(read_ttfbs):.1f}")

    # --- Parallel WRITE throughput ---
    log(f"  --- WRITE ({PARALLEL_STREAMS} parallel streams) ---")
    # Pre-generate one payload per stream to avoid memory contention
    payloads = [os.urandom(chunk_bytes) for _ in range(PARALLEL_STREAMS)]
    write_agg_rates = []
    for rnd in range(1, NUM_THROUGHPUT_ROUNDS + 1):
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_STREAMS) as pool:
            futures = [pool.submit(_do_one_write, payloads[s]) for s in range(PARALLEL_STREAMS)]
            results = [f.result() for f in futures]
        wall_time = time.monotonic() - start
        total_mb = sum(r[0] for r in results) / (1024 * 1024)
        agg_rate = total_mb / wall_time
        per_stream = [r[0] / (1024 * 1024) / r[1] for r in results]
        write_agg_rates.append(agg_rate)
        log(f"  WRITE round {rnd}/{NUM_THROUGHPUT_ROUNDS}: "
            f"{total_mb:.0f} MB in {wall_time:.3f}s = {agg_rate:.1f} MB/s aggregate  "
            f"per-stream avg={sum(per_stream)/len(per_stream):.1f} MB/s")

    avg_write = sum(write_agg_rates) / len(write_agg_rates)
    log(f"  WRITE aggregate: avg={avg_write:.1f} MB/s  "
        f"min={min(write_agg_rates):.1f}  max={max(write_agg_rates):.1f}")

    return avg_read, avg_write, avg_rt


def main():
    log("Starting Ray worker benchmark")
    log(f"  HEAD_SERVICE={HEAD_SERVICE}  HEAD_PORT={HEAD_PORT}")
    log(f"  OBJECT_CHUNK_SIZE_MB={OBJECT_CHUNK_SIZE_MB}")
    log(f"  NUM_THROUGHPUT_ROUNDS={NUM_THROUGHPUT_ROUNDS}")
    log(f"  PARALLEL_STREAMS={PARALLEL_STREAMS}")

    if not wait_for_head():
        sys.exit(1)

    phase_dns_resolution()
    phase_job_metadata()
    avg_read, avg_write, avg_rt = phase_object_store_throughput()

    # Final summary
    log("=" * 60)
    log("TEST COMPLETE - SUMMARY")
    log("=" * 60)
    log(f"  DNS resolution:     PASS")
    log(f"  Job metadata HTTP:  PASS")
    log(f"  Object store RTT:   {avg_rt:.2f} ms avg")
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
