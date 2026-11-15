"""Talk to CodinGame's public API from the command line: sign in once, then run test games,
submit a bot and follow its ranking games.

The API is documented at https://www.codingame.com/api/v1/docs (reference: /docs/api). It acts as
the CodinGame user who approved the token, so every submission made here replaces that user's
arena submission.

    python tools/cg.py login                 # prints a link and a code to approve in the browser
    python tools/cg.py wait                  # waits for the approval and saves the token
    python tools/cg.py me
    python tools/cg.py puzzle                # league, players per game, language ids
    python tools/cg.py test paste/tronzero_paste.rs --opponents boss,boss,boss
    python tools/cg.py submit paste/tronzero_paste.rs
    python tools/cg.py progress
    python tools/cg.py follow                # prints the ranking until all ranking games are played
    python tools/cg.py battles              # win counts of the current submission's ranking games
    python tools/cg.py get puzzles/tron-battle/leaderboard?around_me=true

The token is kept in ~/.config/codingame/token and is never printed. Revoke it at
https://www.codingame.com/settings/tokens. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.codingame.com/api/v1"
HOME = os.path.join(os.path.expanduser("~"), ".config", "codingame")
TOKEN = os.path.join(HOME, "token")
DEVICE = os.path.join(HOME, "device.json")
AGENT = "cg-cli/1.0"
SCOPES = "challenge:read challenge:play challenge:submit"
GRANT = "urn:ietf:params:oauth:grant-type:device_code"
LIMIT_CHARS = 100_000


class ApiError(Exception):
    def __init__(self, status: int, problem: dict):
        self.status = status
        self.problem = problem
        super().__init__(f"{status} {problem.get('title', '')}: {problem.get('detail', '')}")


def write_private(path: str, text: str) -> None:
    os.makedirs(HOME, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def token() -> str:
    try:
        with open(TOKEN, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        sys.exit("No token yet: run `python tools/cg.py login`, approve in the browser, then `wait`.")


def call(method: str, path: str, body=None, auth: bool = True, form: bool = False, timeout: float = 60.0):
    """One request. Returns (status, parsed JSON or text). Raises ApiError on an error status."""
    url = path if path.startswith("http") else API + path
    headers = {"User-Agent": AGENT, "Accept": "application/json, text/plain"}
    data = None
    if body is not None:
        if form:
            data = urllib.parse.urlencode(body).encode("ascii")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
    if auth:
        headers["Authorization"] = "Bearer " + token()
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            kind = resp.headers.get("Content-Type", "")
            return resp.status, (json.loads(raw) if "json" in kind else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            problem = json.loads(raw)
        except ValueError:
            problem = {"title": "error", "detail": raw[:400]}
        if e.code == 429 and "retry_after_seconds" not in problem and e.headers.get("Retry-After"):
            problem["retry_after_seconds"] = int(e.headers["Retry-After"])
        raise ApiError(e.code, problem) from None


def fail(e: ApiError) -> None:
    p = e.problem
    print(f"error {e.status}: {p.get('title', '')}: {p.get('detail', p.get('error_description', ''))}")
    for key in ("code", "hint", "retry_after_seconds"):
        if key in p:
            print(f"  {key}: {p[key]}")
    if e.status == 401:
        print(f"  The token is missing, expired or revoked: delete {TOKEN} and sign in again.")
    sys.exit(1)


def game(args) -> str:
    return f"/challenges/{args.challenge}" if args.challenge else f"/puzzles/{args.puzzle}"


def windows(limits: dict) -> str:
    parts = []
    for kind in ("submits", "test_games", "requests"):
        w = limits.get(kind)
        if w:
            parts.append(f"{w['name']} {w['remaining']}/{w['limit']} left")
    return ", ".join(parts)


def read_code(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        code = f.read()
    units = len(code.encode("utf-16-le")) // 2
    print(f"{path}: {len(code):,} characters ({units:,} UTF-16 units)")
    if units > LIMIT_CHARS:
        sys.exit(f"Over CodinGame's {LIMIT_CHARS:,}-character limit: not sent.")
    return code


# ---- sign-in ---------------------------------------------------------------------------------

def cmd_login(args) -> None:
    if os.path.exists(TOKEN) and not args.force:
        sys.exit(f"A token is already saved in {TOKEN}. Use --force to replace it.")
    _, d = call("POST", "/auth/device", {"client_name": args.name, "scope": args.scope}, auth=False, form=True)
    d["started"] = time.time()
    write_private(DEVICE, json.dumps(d))
    print(f"Open    {d['verification_uri_complete']}")
    print(f"Code    {d['user_code']}")
    print(f"Scopes  {args.scope}")
    print(f"Valid for {d['expires_in'] // 60} minutes. Then run: python tools/cg.py wait")


def cmd_wait(args) -> None:
    try:
        with open(DEVICE, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        sys.exit("No sign-in in progress: run `python tools/cg.py login` first.")
    interval = int(d.get("interval", 5))
    deadline = d["started"] + d["expires_in"]
    while time.time() < deadline:
        time.sleep(interval)
        try:
            _, r = call("POST", "/auth/token", {"grant_type": GRANT, "device_code": d["device_code"]},
                        auth=False, form=True)
        except ApiError as e:
            err = e.problem.get("error", "")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval = int(e.problem.get("interval", interval + 5))
                continue
            os.remove(DEVICE)
            sys.exit(f"Sign-in stopped: {err or e}")
        write_private(TOKEN, r["access_token"])
        os.remove(DEVICE)
        print(f"Token saved. Scopes: {r.get('scope')}. Expires {r.get('expires_at')}.")
        return
    os.remove(DEVICE)
    sys.exit("The code expired before it was approved: run `login` again.")


# ---- reading ---------------------------------------------------------------------------------

def cmd_get(args) -> None:
    # Git Bash rewrites an argument that starts with "/" into a Windows path, so the slash is optional.
    _, r = call("GET", args.path if args.path.startswith(("/", "http")) else "/" + args.path)
    print(r if isinstance(r, str) else json.dumps(r, indent=1, ensure_ascii=False))


def cmd_me(args) -> None:
    _, r = call("GET", "/me")
    print(json.dumps({k: v for k, v in r.items() if k != "limits"}, ensure_ascii=False))


def cmd_puzzle(args) -> None:
    _, r = call("GET", game(args))
    if args.statement:
        print(r.get("statement", "(no statement)"))
        return
    print(f"{r.get('title')}  league {r.get('league')}  players {r.get('players_min')}-{r.get('players_max')}")
    print("languages: " + " ".join(l["id"] for l in r.get("languages", [])))


def cmd_progress(args) -> None:
    _, r = call("GET", game(args) + "/progress")
    keep = ("submission_id", "language", "in_progress", "percentage", "score", "rank", "total", "league",
            "league_rank", "league_total", "eligible_for_promotion", "league_boss", "promotion_mode")
    print(json.dumps({k: r[k] for k in keep if k in r}, ensure_ascii=False))
    if r.get("limits"):
        print("limits: " + windows(r["limits"]))


def cmd_follow(args) -> None:
    """Print the ranking every few minutes until the submission has played all its ranking games."""
    deadline = time.time() + args.minutes * 60
    last = None
    while True:
        _, r = call("GET", game(args) + "/progress")
        line = (f"{r.get('percentage')}%  score {r.get('score')}  rank {r.get('rank')}/{r.get('total')}"
                f"  {r.get('league')} {r.get('league_rank')}/{r.get('league_total')}")
        if line != last:
            print(time.strftime("%H:%M"), line, flush=True)
            last = line
        if not r.get("in_progress"):
            print(f"done: submission {r.get('submission_id')}", flush=True)
            return
        if time.time() > deadline:
            print(f"still ranking after {args.minutes} minutes", flush=True)
            return
        time.sleep(max(30, args.every))


def cmd_submissions(args) -> None:
    _, r = call("GET", game(args) + "/submissions")
    for s in r.get("submissions", [])[: args.limit]:
        print(f"{s['submission_id']}  {s['language']:8s} {s['submitted_at']}{'  current' if s.get('current') else ''}")


def cmd_battles(args) -> None:
    """Every ranking game of a submission: how often it finished first, second, ... by table size."""
    query = {"limit": 100}
    if args.submission:
        query["submission"] = args.submission
    games, before = [], None
    while True:
        q = dict(query, **({"before": before} if before else {}))
        _, r = call("GET", game(args) + "/battles?" + urllib.parse.urlencode(q))
        games += r.get("battles", [])
        before = r.get("next_before")
        if not before or len(games) >= args.max:
            break
    done = [g for g in games if g.get("done")]
    print(f"submission {r.get('submission_id')}: {len(games)} games listed, {len(done)} finished")
    by_size: dict[int, list[int]] = {}
    for g in done:
        n = len(g["players"])
        by_size.setdefault(n, [0] * n)[g["my_position"]] += 1
    for n in sorted(by_size):
        row = by_size[n]
        total = sum(row)
        places = "  ".join(f"{i + 1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} {c}"
                           for i, c in enumerate(row))
        print(f"  {n} players: {total:4d} games  {places}  (first place {100.0 * row[0] / total:.1f}%)")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            for g in games:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


def cmd_limits(args) -> None:
    key = "challenge" if args.challenge else "puzzle"
    _, r = call("GET", f"/limits?{key}={args.challenge or args.puzzle}")
    for kind in ("submits", "test_games"):
        for w in r.get(kind, []):
            print(f"{w['name']:16s} {w['remaining']:4d}/{w['limit']} left")


# ---- playing ---------------------------------------------------------------------------------

def cmd_test(args) -> None:
    body = {"language": args.language, "code": read_code(args.file)}
    if args.replay:
        body["replay_game_id"] = args.replay
    else:
        body["opponents"] = [int(o) if o.lstrip("-").isdigit() else o for o in args.opponents.split(",")]
        if args.options:
            body["game_options"] = args.options.replace(";", "\n")
    try:
        _, r = call("POST", game(args) + "/test-games?wait=30&format=json", body, timeout=90)
        while r.get("status") == "running":
            _, r = call("GET", f"/test-games/{r['test_game_id']}?wait=30&format=json", timeout=90)
    except ApiError as e:
        fail(e)
    if r.get("status") != "done":
        print(f"test game {r.get('status')}: {r.get('failure')}\n  hint: {r.get('hint')}")
        sys.exit(1)
    log = r["log"]
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(log, f, ensure_ascii=False)
    names = {a["index"]: a["name"] + (" (me)" if a.get("me") else "") for a in log.get("agents", [])}
    print(f"game {log.get('game_id')}  options {log.get('game_options', '').splitlines()}")
    print("ranking: " + " > ".join(names.get(i, str(i)) for i in log.get("ranking", [])))
    print(f"scores: {log.get('scores')}  frames: {len(log.get('frames', []))}")
    mine = [f for f in log.get("frames", []) if f.get("agent") == 0]
    errors = [f for f in mine if f.get("error")]
    late = [f for f in log.get("frames", []) if "in time" in (f.get("referee", "") + f.get("summary", ""))
            or "timeout" in f.get("summary", "").lower()]
    print(f"my frames: {len(mine)}  errors: {len(errors)}  timeout lines: {len(late)}"
          + (f"  fail_cause: {log['fail_cause']}" if log.get("fail_cause") else ""))
    for f in (errors + late)[:5]:
        print(f"  frame {f['frame']}: {f.get('error') or f.get('referee') or f.get('summary')}")
    if args.stderr:
        for f in mine[: args.stderr]:
            print(f"  frame {f['frame']}: {f.get('stdout')!r}  {(f.get('stderr') or '').splitlines()[:1]}")


def cmd_submit(args) -> None:
    body = {"language": args.language, "code": read_code(args.file)}
    try:
        status, r = call("POST", game(args) + "/submissions", body, timeout=120)
    except ApiError as e:
        fail(e)
    print(f"submitted ({status}): submission_id {r.get('submission_id')}")
    if r.get("limits"):
        print("limits: " + windows(r["limits"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--puzzle", default="tron-battle", help="pretty_id of a bot programming puzzle")
    ap.add_argument("--challenge", default="", help="public_id of a contest, instead of a puzzle")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("login", help="start the browser sign-in")
    p.add_argument("--name", default="Claude Code", help="shown on the approval page")
    p.add_argument("--scope", default=SCOPES)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_login)
    sub.add_parser("wait", help="wait for the approval and save the token").set_defaults(fn=cmd_wait)

    p = sub.add_parser("get", help="GET any API path and print the answer")
    p.add_argument("path")
    p.set_defaults(fn=cmd_get)
    sub.add_parser("me").set_defaults(fn=cmd_me)
    p = sub.add_parser("puzzle", help="league, players per game and language ids")
    p.add_argument("--statement", action="store_true")
    p.set_defaults(fn=cmd_puzzle)
    sub.add_parser("progress").set_defaults(fn=cmd_progress)
    p = sub.add_parser("follow", help="print the ranking until every ranking game is played")
    p.add_argument("--every", type=int, default=120, help="seconds between checks (30 at least)")
    p.add_argument("--minutes", type=int, default=240, help="give up after this long")
    p.set_defaults(fn=cmd_follow)
    p = sub.add_parser("submissions")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_submissions)
    p = sub.add_parser("battles", help="placements in the ranking games of a submission")
    p.add_argument("--submission", type=int, default=0)
    p.add_argument("--max", type=int, default=2000)
    p.add_argument("--out", default="", help="also write every game as a line of JSON")
    p.set_defaults(fn=cmd_battles)
    sub.add_parser("limits").set_defaults(fn=cmd_limits)

    for name, fn in (("test", cmd_test), ("submit", cmd_submit)):
        p = sub.add_parser(name)
        p.add_argument("file")
        p.add_argument("--language", default="Rust")
        if name == "test":
            p.add_argument("--opponents", default="default", help="comma list: default, boss, me or agent ids")
            p.add_argument("--options", default="", help="referee options, e.g. seed=12 (';' between lines)")
            p.add_argument("--replay", type=int, default=0, help="replay this game id in my seat")
            p.add_argument("--out", default="", help="write the JSON log here")
            p.add_argument("--stderr", type=int, default=0, help="show my first N frames with stderr")
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    # Bot sources and nicknames are not limited to the console's code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        args.fn(args)
    except ApiError as e:
        fail(e)


if __name__ == "__main__":
    main()
