def short(text, width):
    text = "" if text is None else str(text)
    return text if len(text) <= width else text[: width - 3] + "..."


def format_table(entries):
    header = f"{'id':>3}  {'time (UTC)':<10}  {'status':<7}  {'http':>4}  {'pred':>4}  {'prob':>6}  {'ms':>6}  error"
    lines = [header, "-" * len(header)]
    for e in entries:
        prob = "" if e["probability"] is None else f"{e['probability']:.3f}"
        pred = "" if e["prediction"] is None else e["prediction"]
        ms = "" if e["latency_ms"] is None else f"{e['latency_ms']:.1f}"
        lines.append(
            f"{e['id']:>3}  {e['timestamp'][11:19]:<10}  {e['status']:<7}  {e['http_status']:>4}  "
            f"{pred!s:>4}  {prob:>6}  {ms:>6}  {short(e['error'], 60)}"
        )
    return "\n".join(lines)


def format_summary(summary):
    rate = f"{summary['success_rate'] * 100:.0f}%"
    return (
        f"total {summary['total']} | successes {summary['successes']} | failures {summary['failures']} "
        f"| success rate {rate} | failures by status {summary['failures_by_status']}"
    )
