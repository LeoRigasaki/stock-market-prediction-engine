#!/usr/bin/env python3
"""Simple concurrent load test for the prediction API."""

import argparse
import asyncio
import statistics
import time

import aiohttp


async def make_request(session: aiohttp.ClientSession, url: str, token: str, symbols: list[str]) -> float:
    started = time.perf_counter()
    async with session.post(
        f"{url}/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={"symbols": symbols},
    ) as response:
        response.raise_for_status()
        await response.json()
    return time.perf_counter() - started


async def run_load_test(url: str, token: str, concurrency: int, requests_count: int, symbols: list[str]) -> None:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            make_request(session, url, token, symbols)
            for _ in range(requests_count)
        ]
        latencies = await asyncio.gather(*tasks)

    p95_index = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
    sorted_latencies = sorted(latencies)

    print(f"requests: {requests_count}")
    print(f"concurrency: {concurrency}")
    print(f"avg_latency_seconds: {statistics.mean(latencies):.4f}")
    print(f"median_latency_seconds: {statistics.median(latencies):.4f}")
    print(f"p95_latency_seconds: {sorted_latencies[p95_index]:.4f}")
    print(f"max_latency_seconds: {max(latencies):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic concurrent load test against /predict.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base API URL")
    parser.add_argument("--token", default="demo_key_12345", help="Bearer token")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent request count")
    parser.add_argument("--requests", type=int, default=20, help="Total request count")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "NVDA", "MSFT"],
        help="Symbols to include in the payload",
    )
    args = parser.parse_args()

    asyncio.run(
        run_load_test(
            url=args.url.rstrip("/"),
            token=args.token,
            concurrency=args.concurrency,
            requests_count=args.requests,
            symbols=args.symbols,
        )
    )


if __name__ == "__main__":
    main()
