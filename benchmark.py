"""
AI agent memory latency: local SQLite, local ChromaDB, and the MemorySync API.

Every number this script prints is measured on the machine that runs it. A
backend that cannot run (ChromaDB not installed, no network, a refused key) is
reported as not run. Nothing is ever substituted for a missing measurement.

The three backends do different work, so read the numbers as a shape, not a
race:

* SQLite is an in-process keyword filter on an indexed column: no network, no
  ranking by meaning. Microseconds.
* ChromaDB runs in-process too, but embeds the query text locally before a
  vector search. Tens to hundreds of milliseconds, depending on the CPU.
* MemorySync is a remote API: hybrid keyword plus vector ranking, per-user
  scoping and persistence that survives the machine. Two numbers are reported:
  the server's own query time (`latency_ms` in every response) and the full
  round trip from this machine, which is dominated by distance to the API.

Usage:
    pip install -r requirements.txt
    python benchmark.py                       # all three backends
    python benchmark.py --iterations 25
    python benchmark.py --skip-remote         # local backends only (used in CI)

The MemorySync run uses MEMORYSYNC_API_KEY if it is set, and otherwise creates
a free evaluation key with one unauthenticated API call (no signup).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sqlite3
import statistics
import time
import urllib.parse

API = "https://api.memorysync.io"

SEED_FACTS = [
    "The production database runs PostgreSQL 16 in us-east-1.",
    "Deploys are gated on the contract test suite passing in CI.",
    "The team decided against a hosted vector database because of per-namespace pricing.",
    "Tenant isolation is enforced by the database, not by application filters.",
    "Background compaction runs nightly at 02:00 UTC.",
]
QUERIES = [
    "what database do we use",
    "why did we reject the hosted vector database",
    "how is tenant isolation enforced",
    "when does compaction run",
]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Interpolation flatters small samples."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(p / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]


def summary(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
    }


# ---------------------------------------------------------------------------
# Local SQLite
# ---------------------------------------------------------------------------

def benchmark_sqlite(iterations: int) -> dict:
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, tenant_id TEXT, content TEXT, created_at REAL)")
    cur.execute("CREATE INDEX idx_tenant ON memories(tenant_id, created_at)")
    for i in range(500):
        cur.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?)",
            (f"mem_{i}", f"tenant_{i % 10}", f"Fact content for turn {i}", time.time()),
        )
    conn.commit()

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        cur.execute("SELECT content FROM memories WHERE tenant_id = 'tenant_3' ORDER BY created_at DESC LIMIT 5")
        cur.fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)
    conn.close()
    return {"status": "measured", **summary(latencies)}


# ---------------------------------------------------------------------------
# Local ChromaDB
# ---------------------------------------------------------------------------

def benchmark_chroma(iterations: int) -> dict:
    try:
        import chromadb
    except ImportError:
        return {"status": "not run", "reason": "chromadb is not installed (pip install chromadb)"}

    try:
        client = chromadb.Client()
        collection = client.create_collection(f"bench_{int(time.time() * 1000)}")
        docs = [f"Agent decision context for execution turn {i}" for i in range(100)]
        collection.add(documents=docs, ids=[f"id_{i}" for i in range(100)])
        collection.query(query_texts=["warm up"], n_results=5)  # loads the embedding model

        latencies = []
        for i in range(iterations):
            t0 = time.perf_counter()
            collection.query(query_texts=[QUERIES[i % len(QUERIES)]], n_results=5)
            latencies.append((time.perf_counter() - t0) * 1000)
        return {"status": "measured", **summary(latencies)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "not run", "reason": f"{type(exc).__name__}: {exc}"[:200]}


# ---------------------------------------------------------------------------
# MemorySync API
# ---------------------------------------------------------------------------

class _Api:
    """One kept-alive HTTPS connection, the way an SDK's HTTP client behaves.

    Opening a new connection per call adds a TLS handshake to every request,
    which no real client pays after the first call, so it would measure the
    benchmark rather than the service.
    """

    def __init__(self, timeout: int = 45) -> None:
        self._host = urllib.parse.urlsplit(API).netloc
        self._timeout = timeout
        self._conn: http.client.HTTPSConnection | None = None

    def post(self, path: str, payload: dict, headers: dict) -> tuple[int, dict, float]:
        data = json.dumps(payload).encode("utf-8")
        all_headers = {"Content-Type": "application/json", "User-Agent": "memory-benchmarks/2.0", **headers}
        for attempt in (1, 2):
            if self._conn is None:
                self._conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
            t0 = time.perf_counter()
            try:
                self._conn.request("POST", path, body=data, headers=all_headers)
                resp = self._conn.getresponse()
                body = resp.read().decode("utf-8", errors="replace")
            except (http.client.HTTPException, OSError):
                # The server closed an idle connection: reconnect once.
                self._conn.close()
                self._conn = None
                if attempt == 2:
                    raise
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = {}
            return resp.status, parsed if isinstance(parsed, dict) else {}, elapsed
        raise RuntimeError("unreachable")


def benchmark_memorysync(iterations: int) -> dict:
    api = _Api()
    key = os.environ.get("MEMORYSYNC_API_KEY")
    end_user = os.environ.get("MEMORYSYNC_END_USER_ID")
    try:
        if not key:
            status, minted, _ = api.post("/evaluation/keys", {"agent_caller": "memory-benchmarks"}, {})
            key = minted.get("api_key")
            end_user = end_user or minted.get("default_user_id")
            if not key:
                return {"status": "not run", "reason": f"evaluation key refused: HTTP {status}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "not run", "reason": f"could not reach the API: {type(exc).__name__}"}

    headers = {"X-API-Key": key, "X-End-User-ID": end_user or "memory-benchmarks"}

    seeded = 0
    for fact in SEED_FACTS:
        try:
            status, _, _ = api.post("/memory/add", {"text": fact}, headers)
            seeded += status in (200, 201)
        except Exception:  # noqa: BLE001
            pass
    if seeded == 0:
        return {"status": "not run", "reason": "no facts could be stored"}
    time.sleep(3)  # new memories take a moment to become searchable

    round_trips, server_times, failures = [], [], 0
    for i in range(iterations):
        try:
            status, body, elapsed = api.post("/memory/query", {"query": QUERIES[i % len(QUERIES)], "k": 5}, headers)
        except Exception:  # noqa: BLE001
            failures += 1
            continue
        if status != 200:
            failures += 1
            continue
        round_trips.append(elapsed)
        if isinstance(body.get("latency_ms"), (int, float)):
            server_times.append(float(body["latency_ms"]))
        time.sleep(0.15)

    if not round_trips:
        return {"status": "not run", "reason": f"all {failures} queries failed"}
    result = {"status": "measured", "failures": failures, "round_trip": summary(round_trips)}
    if server_times:
        result["server"] = summary(server_times)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fit_labels_inside(fig, ax, notes: list, pad_px: float = 10.0) -> None:
    """Widen the log x-axis just enough that every value label ends inside the plot.

    Matplotlib sizes the axis to the bars, so the label after the longest bar
    used to run across the frame. On a log axis a label's width in data units
    depends on the limits themselves, so the right limit is solved from the
    labels' measured pixel widths instead of a guessed factor.
    """
    import math

    for _ in range(5):
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = ax.get_window_extent(renderer)
        left, right = ax.get_xlim()
        lo, hi = math.log10(left), math.log10(right)
        needed = hi
        for note in notes:
            anchor_px = ax.transData.transform(note.xy)[0]
            reach_px = note.get_window_extent(renderer).x1 - anchor_px  # offset plus text width
            usable = box.width - reach_px - pad_px
            if usable > 0:
                needed = max(needed, lo + (math.log10(note.xy[0]) - lo) * box.width / usable)
        if needed <= hi + 1e-6:
            return
        ax.set_xlim(left, 10**needed)


def render_chart(results: dict, path: str) -> None:
    """One chart, drawn only from what this run measured."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    if results["sqlite"]["status"] == "measured":
        rows.append(("SQLite (local)", results["sqlite"]))
    if results["chromadb"]["status"] == "measured":
        rows.append(("ChromaDB (local)", results["chromadb"]))
    ms = results["memorysync"]
    if ms["status"] == "measured":
        if "server" in ms:
            rows.append(("MemorySync (server time)", ms["server"]))
        rows.append(("MemorySync (round trip)", ms["round_trip"]))

    labels = [r[0] for r in rows]
    p50 = [max(r[1]["p50_ms"], 0.001) for r in rows]
    p95 = [max(r[1]["p95_ms"], 0.001) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="white")
    y = range(len(rows))
    ax.barh([i + 0.2 for i in y], p50, height=0.38, color="#2563eb", label="p50")
    ax.barh([i - 0.2 for i in y], p95, height=0.38, color="#93c5fd", label="p95")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_title("Measured retrieval latency", fontweight="bold")

    def label(v: float) -> str:
        if v >= 10:
            return f"{v:,.0f} ms"
        if v >= 1:
            return f"{v:.1f} ms"
        return f"{v:.3f} ms"

    # Each value sits a fixed 4 points past the end of its bar. An offset in
    # points keeps the gap the same however far the axis is widened below.
    notes = []
    for i, (a, b) in enumerate(zip(p50, p95)):
        notes.append(ax.annotate(label(a), xy=(a, i + 0.2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9))
        notes.append(ax.annotate(label(b), xy=(b, i - 0.2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9))
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", which="both", ls=":", alpha=0.5)
    _fit_labels_inside(fig, ax, notes)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-remote", action="store_true", help="measure the local backends only")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    results = {
        "iterations": args.iterations,
        "sqlite": benchmark_sqlite(args.iterations),
        "chromadb": benchmark_chroma(args.iterations),
        "memorysync": {"status": "not run", "reason": "--skip-remote"}
        if args.skip_remote
        else benchmark_memorysync(args.iterations),
    }

    def line(name: str, r: dict) -> str:
        if r.get("status") != "measured":
            return f"{name:<28} not run: {r.get('reason')}"
        return f"{name:<28} p50 {r['p50_ms']:>10.3f} ms   p95 {r['p95_ms']:>10.3f} ms   ({r['samples']} samples)"

    print(line("SQLite (local)", results["sqlite"]))
    print(line("ChromaDB (local)", results["chromadb"]))
    ms = results["memorysync"]
    if ms["status"] == "measured":
        if "server" in ms:
            print(line("MemorySync (server time)", {"status": "measured", **ms["server"]}))
        print(line("MemorySync (round trip)", {"status": "measured", **ms["round_trip"]}))
    else:
        print(line("MemorySync", ms))

    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    if not args.no_chart:
        render_chart(results, os.path.join("assets", "measured_latency.png"))
        print("wrote results.json and assets/measured_latency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
