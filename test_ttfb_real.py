#!/usr/bin/env python3
"""
Real TTFB test - measures time until actual content (not just HTTP headers)
"""
import time
import requests

def test_real_ttfb(url, model, message):
    """Measure time until first actual content chunk"""
    print(f"\nTesting {model} at {url}")
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-OpenWebUI-Chat-Id": f"ttfb-real-{int(time.time())}",
    }
    
    start = time.time()
    print(f"Request sent: {start:.3f}")
    
    response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
    
    first_data = None
    for i, line in enumerate(response.iter_lines(decode_unicode=True)):
        if not line or line.startswith('data: [DONE]'):
            continue
            
        if line.startswith('data: '):
            # Got actual SSE data
            if first_data is None:
                first_data = time.time()
                ttfb = first_data - start
                print(f"✓ First content: {ttfb:.3f}s ({ttfb*1000:.0f}ms)")
                print(f"  Line {i}: {line[:100]}")
                return ttfb
                
    return None

# Test both dev (no MCP) and prod (with MCP)
tests = [
    ("Dev (no MCP)", "http://192.168.3.142:9000/v1/chat/completions", "claude-cli"),
    ("Prod (with MCP)", "http://192.168.3.142:8000/v1/chat/completions", "claude-cli"),
]

results = []
for name, url, model in tests:
    ttfb = test_real_ttfb(url, model, "Say hello")
    if ttfb:
        results.append((name, ttfb))
    time.sleep(2)

print("\n" + "="*60)
print("REAL TTFB RESULTS (Time to actual content)")
print("="*60)
for name, ttfb in results:
    print(f"{name:30s} {ttfb:6.3f}s ({ttfb*1000:6.0f}ms)")
