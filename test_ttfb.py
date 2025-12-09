#!/usr/bin/env python3
"""
TTFB (Time To First Byte) test for Claude Bridge
Tests how quickly the bridge starts streaming responses
"""

import time
import requests
import json

# Test configurations
TESTS = [
    {
        "name": "Dev Bridge - Simple Question",
        "url": "http://192.168.3.142:9000/v1/chat/completions",
        "model": "claude-cli",
        "message": "Say 'hello' in one word",
    },
    {
        "name": "Prod Bridge - Simple Question",
        "url": "http://192.168.3.142:8000/v1/chat/completions",
        "model": "claude-cli",
        "message": "Say 'hello' in one word",
    },
    {
        "name": "Dev Bridge - Codex",
        "url": "http://192.168.3.142:9000/v1/chat/completions",
        "model": "codex-cli",
        "message": "Say 'hello' in one word",
    },
    {
        "name": "Dev Bridge - Gemini",
        "url": "http://192.168.3.142:9000/v1/chat/completions",
        "model": "gemini-cli",
        "message": "Say 'hello' in one word",
    },
]


def test_ttfb(name, url, model, message):
    """Test TTFB for a single request"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "X-OpenWebUI-Chat-Id": f"ttfb-test-{int(time.time())}",
    }

    try:
        start_time = time.time()
        print(f"Request sent at: {start_time:.3f}")

        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)

        first_byte_time = None
        chunk_count = 0
        total_bytes = 0

        for chunk in response.iter_content(chunk_size=1):
            if first_byte_time is None:
                first_byte_time = time.time()
                ttfb = first_byte_time - start_time
                print(f"\n✓ First byte received!")
                print(f"  TTFB: {ttfb:.3f} seconds ({ttfb*1000:.0f}ms)")

            chunk_count += 1
            total_bytes += len(chunk)

            # Stop after getting first data chunk (not just headers)
            if chunk_count > 100:  # Get ~100 bytes to ensure we got real data
                break

        if first_byte_time:
            print(f"  Chunks received: {chunk_count}")
            print(f"  Total bytes: {total_bytes}")
            return ttfb
        else:
            print("✗ No data received")
            return None

    except requests.exceptions.Timeout:
        print("✗ Request timed out (30s)")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Run all TTFB tests"""
    print("\n" + "="*60)
    print("Claude Bridge TTFB Test Suite")
    print("="*60)

    results = []

    for test in TESTS:
        ttfb = test_ttfb(
            name=test["name"],
            url=test["url"],
            model=test["model"],
            message=test["message"],
        )

        if ttfb is not None:
            results.append((test["name"], ttfb))

        time.sleep(2)  # Brief pause between tests

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        for name, ttfb in results:
            print(f"{name:40s} {ttfb:6.3f}s ({ttfb*1000:6.0f}ms)")

        avg_ttfb = sum(t for _, t in results) / len(results)
        print(f"\nAverage TTFB: {avg_ttfb:.3f}s ({avg_ttfb*1000:.0f}ms)")
    else:
        print("No successful tests")


if __name__ == "__main__":
    main()
