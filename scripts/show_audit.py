import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.audit_log import AuditLog
from audit_table import format_summary, format_table


def main():
    parser = argparse.ArgumentParser(description="Print the audit trail straight from the SQLite file")
    parser.add_argument("--db", default=str(config.AUDIT_DB_PATH), help="path to audit.db")
    parser.add_argument("--limit", type=int, default=20, help="how many rows to show")
    parser.add_argument("--status", choices=["success", "failure"], help="only show this kind")
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"No audit database at {args.db}. Send a request to /predict first.")

    log = AuditLog(args.db)
    print(format_table(list(reversed(log.recent(args.limit, args.status)))))
    print()
    print(format_summary(log.summary()))


if __name__ == "__main__":
    main()
