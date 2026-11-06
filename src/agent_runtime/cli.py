"""Command line: agentrt demo | verify | show | replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .demo import run_demo
from .policy import Policy, policy_from_dict
from .replay import replay
from .trace import read_events, verify

OK, FAILED, USAGE = 0, 1, 2


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else None
    for title, report in run_demo(out):
        extra = f", {report.denied} refused, {report.injections} injection(s) flagged" if report.denied or report.injections else ""
        print(f"{title}: {report.status}{extra}")
        print(f"  {report.answer or report.reason}")
    if out:
        print(f"\nTraces written to {out}. Try: agentrt show {out / 'demo-2.jsonl'}")
    return OK


def cmd_verify(args: argparse.Namespace) -> int:
    events = read_events(Path(args.trace))
    problems = verify(events)
    if problems:
        print(f"TAMPERED or damaged: {len(problems)} problem(s)", file=sys.stderr)
        for line in problems[:20]:
            print(f"  {line}", file=sys.stderr)
        return FAILED
    print(f"OK: {len(events)} events, chain intact. Final hash {events[-1]['hash'] if events else '-'}")
    return OK


def cmd_show(args: argparse.Namespace) -> int:
    for event in read_events(Path(args.trace)):
        data = event["data"]
        kind = event["kind"]
        if kind == "run_start":
            line = f"task: {data['task']}"
        elif kind == "decision":
            line = "model: final answer" if "final" in data else f"model: {data['tool']}({json.dumps(data.get('args', {}))[:70]})"
        elif kind == "verdict":
            line = f"policy: {data['action'].upper()} {data['tool']}" + (f"  - {data['reason']}" if data.get("reason") else "")
        elif kind == "approval":
            line = f"human: {'approved' if data['approved'] else 'refused'} {data['tool']}"
        elif kind == "tool_result":
            line = f"tool: {data['tool']} {'ok' if data['ok'] else 'FAILED'}"
        elif kind == "injection_suspected":
            line = f"ALERT: injection signals from {data['tool']}: {', '.join(data['signals'])}"
        elif kind == "run_end":
            line = f"end: {data['status']}" + (f" - {data['reason']}" if data.get("reason") else "")
        else:
            line = kind
        print(f"{event['seq']:>3}  {line}")
    return OK


def cmd_replay(args: argparse.Namespace) -> int:
    events = read_events(Path(args.trace))
    problems = verify(events)
    if problems:
        print("Refusing to replay a trace whose chain is broken (run 'agentrt verify').", file=sys.stderr)
        return FAILED
    policy = policy_from_dict(json.loads(Path(args.policy).read_text(encoding="utf-8"))) if args.policy else Policy()
    result = replay(events, policy, approve=False if args.refuse_asks else None)
    print(f"Compared {result.compared} verdict(s) against the recording.")
    for change in result.changes:
        print(f"  {change}")
    if not result.changes:
        print("  No verdict changed.")
    print(f"Original: {result.original.status}   Replayed: {result.replayed.status}")
    return FAILED if (args.fail_on_change and result.changes) else OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentrt", description="An agent runtime where safety is code: policy, approvals, budgets, audit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run three scripted scenarios (no model, no network)")
    demo.add_argument("--out", help="write each run's trace into this directory")
    demo.set_defaults(func=cmd_demo)
    ver = sub.add_parser("verify", help="check a trace's hash chain")
    ver.add_argument("trace")
    ver.set_defaults(func=cmd_verify)
    show = sub.add_parser("show", help="print a trace as a readable timeline")
    show.add_argument("trace")
    show.set_defaults(func=cmd_show)
    rep = sub.add_parser("replay", help="replay a recorded run against a policy and list the verdicts that change")
    rep.add_argument("trace")
    rep.add_argument("--policy", help="policy JSON file (default: the built-in defaults)")
    rep.add_argument("--refuse-asks", action="store_true", help="answer every approval request with no")
    rep.add_argument("--fail-on-change", action="store_true")
    rep.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE


if __name__ == "__main__":
    raise SystemExit(main())
