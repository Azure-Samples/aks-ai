"""
Ray Worker Node Simulator - Tests performance with the head service.

Performs three test phases:
  1. DNS Resolution  - Resolve the head service hostname, measure latency
  2. Job Metadata    - Fetch job list, retrieve details, update status via HTTP
  3. iperf3 TCP      - Measure network throughput and RTT to the head node
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

HEAD_SERVICE = os.environ.get("HEAD_SERVICE", "ray-head-svc")
HEAD_PORT = int(os.environ.get("HEAD_PORT", "8265"))
IPERF_PORT = int(os.environ.get("IPERF_PORT", "5201"))
HEAD_IP = None  # Resolved once at startup to bypass DNS on every request
WORKER_ID = os.environ.get("HOSTNAME", "worker-unknown")
IPERF_DURATION = int(os.environ.get("IPERF_DURATION", "60"))
IPERF_PARALLEL = int(os.environ.get("IPERF_PARALLEL", "32"))
IPERF_LENGTH = os.environ.get("IPERF_LENGTH", "128K")
RETRY_INTERVAL = 5
MAX_RETRIES = 24  # 2 minutes


def log(msg):
    print(f"[WORKER {WORKER_ID}] {msg}", flush=True)


def percentile(values, p):
    """Return the p-th percentile (0-100) using nearest-rank."""
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


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
        log(f"  DNS latency ({len(latencies)} lookups): "
            f"p50={percentile(latencies, 50):.2f} ms  "
            f"p90={percentile(latencies, 90):.2f} ms  "
            f"p99={percentile(latencies, 99):.2f} ms")
    else:
        log("  DNS latency test: all lookups failed")
    return True


def _host():
    """Return the resolved head IP if available, otherwise the service name."""
    return HEAD_IP or HEAD_SERVICE


def http_get(path):
    url = f"http://{_host()}:{HEAD_PORT}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def http_post(path, data):
    url = f"http://{_host()}:{HEAD_PORT}{path}"
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


def _run_iperf(direction, reverse=False):
    """Run a single iperf3 test and return parsed JSON result.

    Retries up to 5 times if the server is busy (another client connected).

    Args:
        direction: Label for logging ("send" or "receive").
        reverse: If True, run in reverse mode (server sends to client).
    """
    cmd = [
        "iperf3",
        "--client", _host(),
        "--port", str(IPERF_PORT),
        "--time", str(IPERF_DURATION),
        "--parallel", str(IPERF_PARALLEL),
        "--length", IPERF_LENGTH,
        "--json",
    ]
    if reverse:
        cmd.append("--reverse")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        log(f"  Running iperf3 {direction} test (attempt {attempt}): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=IPERF_DURATION + 30)

        if result.returncode != 0:
            # iperf3 --json puts error details in stdout, not stderr
            error_msg = result.stderr.strip()
            try:
                err_json = json.loads(result.stdout)
                error_msg = err_json.get("error", result.stdout[:500])
            except (json.JSONDecodeError, ValueError):
                error_msg = error_msg or result.stdout[:500]
            log(f"  iperf3 {direction} FAILED (rc={result.returncode}): {error_msg}")
            if attempt < max_attempts:
                log(f"  Retrying in 5s...")
                time.sleep(5)
                continue
            return None

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log(f"  iperf3 {direction} JSON parse error: {e}")
            log(f"  stdout: {result.stdout[:500]}")
            return None

    return None


def _parse_iperf_results(data, direction):
    """Extract and log key metrics from iperf3 JSON output.

    Returns (throughput_gbps, rtt_ms) or None on failure.
    """
    if data is None:
        return None

    end = data.get("end", {})

    # --- Per-interval RTT (from stream-level results) ---
    intervals = data.get("intervals", [])
    per_interval_rtts = []
    for interval in intervals:
        for stream in interval.get("streams", []):
            rtt_us = stream.get("rtt")
            if rtt_us is not None and rtt_us > 0:
                per_interval_rtts.append(rtt_us / 1000.0)  # us -> ms

    # --- Aggregate throughput ---
    sum_sent = end.get("sum_sent", {})
    sum_received = end.get("sum_received", {})
    sent_bps = sum_sent.get("bits_per_second", 0)
    recv_bps = sum_received.get("bits_per_second", 0)
    sent_gbps = sent_bps / 1e9
    recv_gbps = recv_bps / 1e9
    retransmits = sum_sent.get("retransmits", "N/A")

    log(f"  {direction} throughput: sent={sent_gbps:.2f} Gbps  "
        f"received={recv_gbps:.2f} Gbps  retransmits={retransmits}")

    # --- Per-stream summary ---
    for stream in end.get("streams", []):
        s = stream.get("sender", {})
        sid = s.get("socket", "?")
        sbps = s.get("bits_per_second", 0) / 1e9
        log(f"    stream {sid}: {sbps:.2f} Gbps")

    # --- RTT percentiles from per-interval data ---
    if per_interval_rtts:
        log(f"  {direction} RTT ({len(per_interval_rtts)} samples): "
            f"p50={percentile(per_interval_rtts, 50):.3f} ms  "
            f"p90={percentile(per_interval_rtts, 90):.3f} ms  "
            f"p99={percentile(per_interval_rtts, 99):.3f} ms")
    else:
        # Fallback: use the mean_rtt from per-stream end summary
        stream_rtts = []
        for stream in end.get("streams", []):
            sender = stream.get("sender", {})
            mean_rtt = sender.get("mean_rtt")
            if mean_rtt is not None and mean_rtt > 0:
                stream_rtts.append(mean_rtt / 1000.0)  # us -> ms
        if stream_rtts:
            log(f"  {direction} RTT (stream means): "
                f"p50={percentile(stream_rtts, 50):.3f} ms  "
                f"p90={percentile(stream_rtts, 90):.3f} ms  "
                f"p99={percentile(stream_rtts, 99):.3f} ms")
        else:
            log(f"  {direction} RTT: not available")

    throughput_gbps = recv_gbps
    rtt_ms = percentile(per_interval_rtts, 50) if per_interval_rtts else None
    return throughput_gbps, rtt_ms


def phase_iperf():
    """Phase 3: Measure network throughput and RTT using iperf3 TCP."""
    log("=" * 60)
    log("PHASE 3: iperf3 TCP Network Benchmark")
    log(f"  target={_host()}:{IPERF_PORT}  duration={IPERF_DURATION}s  "
        f"parallel={IPERF_PARALLEL}  length={IPERF_LENGTH}")
    log("=" * 60)

    # --- Send test (client -> server) ---
    log("  --- Send (client -> server) ---")
    send_data = _run_iperf("send", reverse=False)
    send_result = _parse_iperf_results(send_data, "SEND")

    # Small delay between tests to let iperf3 server restart
    time.sleep(2)

    # --- Receive test (server -> client, reverse mode) ---
    log("  --- Receive (server -> client) ---")
    recv_data = _run_iperf("receive", reverse=True)
    recv_result = _parse_iperf_results(recv_data, "RECV")

    return send_result, recv_result


def main():
    log("Starting Ray worker benchmark")
    log(f"  HEAD_SERVICE={HEAD_SERVICE}  HEAD_PORT={HEAD_PORT}")
    log(f"  IPERF_PORT={IPERF_PORT}  IPERF_DURATION={IPERF_DURATION}s")
    log(f"  IPERF_PARALLEL={IPERF_PARALLEL}  "
        f"IPERF_LENGTH={IPERF_LENGTH}")

    if not wait_for_head():
        sys.exit(1)

    phase_dns_resolution()

    # Resolve head pod IP so subsequent phases skip DNS on every request
    global HEAD_IP
    try:
        results = socket.getaddrinfo(HEAD_SERVICE, HEAD_PORT, socket.AF_INET, socket.SOCK_STREAM)
        HEAD_IP = results[0][4][0]
        log(f"Using head pod IP: {HEAD_IP} (bypassing DNS for benchmarks)")
    except socket.gaierror as e:
        log(f"WARNING: could not resolve head IP, falling back to service name: {e}")

    phase_job_metadata()
    send_result, recv_result = phase_iperf()

    # Final summary
    log("=" * 60)
    log("TEST COMPLETE - SUMMARY")
    log("=" * 60)
    log(f"  DNS resolution:     PASS")
    log(f"  Job metadata HTTP:  PASS")

    if send_result:
        send_tp, send_rtt = send_result
        rtt_str = f"  RTT p50={send_rtt:.3f} ms" if send_rtt else ""
        log(f"  iperf3 SEND:        {send_tp:.2f} Gbps{rtt_str}")
    else:
        log(f"  iperf3 SEND:        FAILED")

    if recv_result:
        recv_tp, recv_rtt = recv_result
        rtt_str = f"  RTT p50={recv_rtt:.3f} ms" if recv_rtt else ""
        log(f"  iperf3 RECV:        {recv_tp:.2f} Gbps{rtt_str}")
    else:
        log(f"  iperf3 RECV:        FAILED")

    # Mark job as complete
    http_post(f"/api/jobs/job-001/status", {
        "status": "SUCCEEDED",
        "worker_id": WORKER_ID,
    })
    log("Done.")


if __name__ == "__main__":
    main()
