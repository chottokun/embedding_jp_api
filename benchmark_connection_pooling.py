import time
import httpx
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket


# A simple HTTP server that returns 200 OK with some JSON
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging of each request to avoid polluting stdout
        pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "message": "hello from mock TEI"}')


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_server(server):
    server.serve_forever()


def run_baseline(url, count):
    start = time.perf_counter()
    for _ in range(count):
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"test": "data"})
            assert resp.status_code == 200
    return time.perf_counter() - start


def run_optimized(url, count):
    start = time.perf_counter()
    shared_client = httpx.Client(timeout=30.0)
    for _ in range(count):
        resp = shared_client.post(url, json={"test": "data"})
        assert resp.status_code == 200
    shared_client.close()
    return time.perf_counter() - start


def run_concurrent_baseline(url, threads_count, reqs_per_thread):
    def worker():
        for _ in range(reqs_per_thread):
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json={"test": "data"})
                assert resp.status_code == 200

    threads = []
    start = time.perf_counter()
    for _ in range(threads_count):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def run_concurrent_optimized(url, threads_count, reqs_per_thread):
    shared_client = httpx.Client(timeout=30.0)

    def worker():
        for _ in range(reqs_per_thread):
            resp = shared_client.post(url, json={"test": "data"})
            assert resp.status_code == 200

    threads = []
    start = time.perf_counter()
    for _ in range(threads_count):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    shared_client.close()
    return time.perf_counter() - start


def main():
    port = get_free_port()
    server = HTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}/"
    print(f"Mock server running at {url}")

    # Warmup
    with httpx.Client() as client:
        client.post(url, json={})

    count = 100
    print(f"Running sequential benchmark ({count} requests)...")
    baseline_seq = run_baseline(url, count)
    optimized_seq = run_optimized(url, count)
    improvement_seq = (baseline_seq - optimized_seq) / baseline_seq * 100
    print(f"Sequential - Baseline (Recreating Client): {baseline_seq:.4f}s")
    print(f"Sequential - Optimized (Shared Client):    {optimized_seq:.4f}s")
    print(
        f"Sequential Speedup: {baseline_seq / optimized_seq:.2f}x ({improvement_seq:.2f}% faster)"
    )

    threads_count = 10
    reqs_per_thread = 10
    total_reqs = threads_count * reqs_per_thread
    print(
        f"\nRunning concurrent benchmark ({total_reqs} total requests across {threads_count} threads)..."
    )
    baseline_con = run_concurrent_baseline(url, threads_count, reqs_per_thread)
    optimized_con = run_concurrent_optimized(url, threads_count, reqs_per_thread)
    improvement_con = (baseline_con - optimized_con) / baseline_con * 100
    print(f"Concurrent - Baseline (Recreating Client): {baseline_con:.4f}s")
    print(f"Concurrent - Optimized (Shared Client):    {optimized_con:.4f}s")
    print(
        f"Concurrent Speedup: {baseline_con / optimized_con:.2f}x ({improvement_con:.2f}% faster)"
    )

    server.shutdown()
    server.server_close()


if __name__ == "__main__":
    main()
