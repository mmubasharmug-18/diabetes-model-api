import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from audit_table import format_summary, format_table

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def load(name):
    return (SAMPLES / name).read_bytes()


def call(method, path, body=None):
    request = urllib.request.Request(BASE + path, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    except urllib.error.URLError:
        raise SystemExit(f"Could not reach {BASE}. Is the server running?")


STEPS = [
    ("valid patient 1", load("valid.json")),
    ("valid patient 2", load("valid_2.json")),
    ("valid patient 3", load("valid_3.json")),
    ("glucose and age out of range", load("invalid_range.json")),
    ("wrong data types and a missing field", load("invalid_types.json")),
    ("broken JSON", b"{this is not json"),
]


def main():
    print("Sending requests to", BASE + "/predict")
    print()
    for label, body in STEPS:
        status, reply = call("POST", "/predict", body)
        outcome = reply.get("label") or "rejected"
        print(f"{label:<40} -> HTTP {status} ({outcome})")

    print()
    print("Audit trail (oldest first):")
    print()
    status, entries = call("GET", "/audit?limit=%d" % len(STEPS))
    print(format_table(list(reversed(entries))))
    print()
    status, summary = call("GET", "/audit/summary")
    print(format_summary(summary))


if __name__ == "__main__":
    main()
