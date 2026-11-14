"""Build the single-file CodinGame paste from the engine sources and a network.

    python tools/make_paste.py --net runs/n1/nets/0040.bin [--bits 7] [--small-bits 12] [--out paste]

Writes into the output folder:
  tronzero_paste.rs   the file to paste into CodinGame (Rust)
  net_packed.txt      the weights as the paste embeds them (`tronzero --packed`)
  net_q.bin           the same weights as a float32 file: exactly what the
                      paste decodes, for measuring what quantisation costs

The paste is `engine/src/{board,net,mcts,play}.rs`, each wrapped in a module,
with comments, indentation and the regions between `// paste:skip-begin` and
`// paste:skip-end` removed, followed by the weights as one string literal.

Weights: every weight tensor is rounded to a few bits per weight with one
scale per output channel: `--bits` for the hidden layer, which holds nine
tenths of the weights, `--small-bits` for the rest (the convolutions, the
scalar features and the two heads, which need finer steps: at 8 bits they
move some value outputs by half the value range). Biases and scales stay
float32. The bit stream is written 14 bits per character as CJK ideographs
(U+4E00..U+8DFF). CodinGame's 100,000 limit counts characters, so a 7-bit
weight costs half a character.

Needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import os
import re
import struct

NPLANES, NSCAL, NVAL = 16, 13, 4
CHAR_BASE, CHAR_BITS = 0x4E00, 14
LIMIT = 100_000
MODULES = ["board", "net", "mcts", "play"]


def f32(x: float) -> float:
    """Round a Python float to the nearest float32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def read_net(path: str) -> dict:
    data = open(path, "rb").read()
    assert data[:4] == b"TZN2", "not a TZN2 weight file"
    planes, scal, c1, c2, c3, hid, nval = struct.unpack_from("<7I", data, 4)
    assert (planes, scal, nval) == (NPLANES, NSCAL, NVAL)
    pos = 32

    def take(n: int) -> list[float]:
        nonlocal pos
        out = list(struct.unpack_from(f"<{n}f", data, pos))
        pos += 4 * n
        return out

    c = [c1, c2, c3]
    cin = [NPLANES, c1, c2]
    net = {"c": c, "hid": hid, "tensors": []}
    # (name, values, number of scale groups, group is the last index, bias)
    for i in range(3):
        w = take(9 * cin[i] * c[i])
        net["tensors"].append((f"conv{i}", w, c[i], True, take(c[i])))
    w = take(8 * 5 * c3 * hid)
    net["tensors"].append(("hidden", w, hid, True, take(hid)))
    net["tensors"].append(("feat", take(NSCAL * hid), hid, True, None))
    w = take(4 * hid)
    net["tensors"].append(("policy", w, 4, False, take(4)))
    w = take(NVAL * hid)
    net["tensors"].append(("value", w, NVAL, False, take(NVAL)))
    assert pos == len(data), "trailing bytes in weight file"
    return net


class BitWriter:
    def __init__(self):
        self.acc = 0
        self.n = 0
        self.chars: list[str] = []

    def put(self, value: int, bits: int) -> None:
        assert 0 <= value < (1 << bits)
        self.acc = (self.acc << bits) | value
        self.n += bits
        while self.n >= CHAR_BITS:
            self.n -= CHAR_BITS
            self.chars.append(chr(CHAR_BASE + ((self.acc >> self.n) & ((1 << CHAR_BITS) - 1))))
        self.acc &= (1 << self.n) - 1

    def f32(self, x: float) -> None:
        self.put(struct.unpack("<I", struct.pack("<f", x))[0], 32)

    def text(self) -> str:
        if self.n:
            self.put(0, CHAR_BITS - self.n)
        return "".join(self.chars)


def quantise(net: dict, bits_of: dict) -> tuple[str, dict, dict]:
    """The packed text, the network it decodes to, and error statistics.
    `bits_of` gives the bits per weight of every tensor, by name."""
    out = BitWriter()
    for c in net["c"]:
        out.put(c, 8)
    out.put(net["hid"], 8)
    decoded = {"c": net["c"], "hid": net["hid"], "tensors": []}
    stats = {}
    for name, w, groups, inner, bias in net["tensors"]:
        bits = bits_of[name]
        qmax = (1 << (bits - 1)) - 1
        zero = 1 << (bits - 1)
        out.put(bits, 8)
        n = len(w)
        group = (lambda i: i % groups) if inner else (lambda i: i // (n // groups))
        peak = [0.0] * groups
        for i, x in enumerate(w):
            g = group(i)
            peak[g] = max(peak[g], abs(x))
        scale = [f32(p / qmax) if p > 0 else 1.0 for p in peak]
        for s in scale:
            out.f32(s)
        deq = []
        err2 = sig2 = 0.0
        for i, x in enumerate(w):
            s = scale[group(i)]
            q = max(-qmax, min(qmax, round(x / s)))
            out.put(q + zero, bits)
            y = f32(q * s)  # the float32 product, as the engine computes it
            deq.append(y)
            err2 += (x - y) ** 2
            sig2 += x * x
        if bias is not None:
            for b in bias:
                out.f32(b)
        decoded["tensors"].append((name, deq, groups, inner, bias))
        stats[name] = (err2 / max(sig2, 1e-30)) ** 0.5
    return out.text(), decoded, stats


def write_net(net: dict, path: str) -> None:
    with open(path, "wb") as f:
        f.write(b"TZN2")
        f.write(struct.pack("<7I", NPLANES, NSCAL, *net["c"], net["hid"], NVAL))
        for _, w, _, _, bias in net["tensors"]:
            f.write(struct.pack(f"<{len(w)}f", *w))
            if bias is not None:
                f.write(struct.pack(f"<{len(bias)}f", *bias))


def strip_line(line: str) -> str:
    """A source line without its `//` comment and surrounding whitespace."""
    i, n = 0, len(line)
    in_string = False
    while i < n:
        ch = line[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "'":
            # a char literal ('x' or '\x'); a lifetime has no closing quote there
            if i + 2 < n and line[i + 2] == "'":
                i += 2
            elif i + 3 < n and line[i + 1] == "\\" and line[i + 3] == "'":
                i += 3
        elif ch == "/" and i + 1 < n and line[i + 1] == "/":
            line = line[:i]
            break
        i += 1
    assert not in_string, f"string literal spans lines: {line!r}"
    return line.strip()


def module_source(path: str) -> str:
    out = []
    skipping = False
    for raw in open(path, encoding="utf-8").read().split("\n"):
        marker = raw.strip()
        if marker == "// paste:skip-begin":
            assert not skipping, path
            skipping = True
            continue
        if marker == "// paste:skip-end":
            assert skipping, path
            skipping = False
            continue
        if skipping:
            continue
        line = strip_line(raw)
        if not line or line.startswith("#[allow(clippy::"):
            continue
        out.append(line)
    assert not skipping, path
    return "\n".join(out)


HEADER = """// tronzero: an AlphaZero-style agent for CodinGame Tron Battle, 2 to 4 players.
// Generated by tools/make_paste.py from engine/src ({modules}) and {net}.
// Change the sources and regenerate; do not edit this file, except the two budgets below.
// The last lines hold the network: {params} weights, {bits} bits each in the hidden layer and
// {small} in the others, 14 bits per character.
#![allow(dead_code, unused_imports, unused_variables, unused_mut, unused_unsafe)]
// Milliseconds to think per turn (CodinGame allows 100). Lower them if a turn ever times out.
pub const TURN_BUDGET_MS: u64 = {turn};
pub const FIRST_TURN_BUDGET_MS: u64 = {first};
"""

MAIN = """fn main() {{
let started = std::time::Instant::now();
let net = net::Net::from_packed(NET);
play::run(&net, &mcts::Params {{ c_puct: {c_puct}, fpu_reduction: {fpu} }}, started);
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True, help="float32 TZN2 weight file")
    ap.add_argument("--bits", type=int, default=7, help="bits per weight in the hidden layer (2..16)")
    ap.add_argument("--small-bits", type=int, default=12, help="bits per weight elsewhere (2..16)")
    ap.add_argument("--out", default="paste", help="output folder")
    ap.add_argument("--src", default="engine/src")
    ap.add_argument("--turn-ms", type=int, default=95)
    ap.add_argument("--first-turn-ms", type=int, default=80)
    ap.add_argument("--line", type=int, default=250, help="characters of weights per source line")
    args = ap.parse_args()
    assert 2 <= args.bits <= 16 and 2 <= args.small_bits <= 16

    net = read_net(args.net)
    bits_of = {name: args.bits if name == "hidden" else args.small_bits for name, *_ in net["tensors"]}
    text, decoded, stats = quantise(net, bits_of)
    os.makedirs(args.out, exist_ok=True)
    write_net(decoded, os.path.join(args.out, "net_q.bin"))
    with open(os.path.join(args.out, "net_packed.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")

    n_params = sum(len(w) + (len(b) if b else 0) for _, w, _, _, b in net["tensors"])
    parts = [
        HEADER.format(
            modules=", ".join(MODULES), net=args.net.replace("\\", "/"), params=n_params,
            bits=args.bits, small=args.small_bits, turn=args.turn_ms, first=args.first_turn_ms,
        )
    ]
    for m in MODULES:
        parts.append(f"mod {m} {{\n{module_source(os.path.join(args.src, m + '.rs'))}\n}}\n")
    # The search settings are the engine's: read them from its source.
    main_rs = open(os.path.join(args.src, "main.rs"), encoding="utf-8").read()
    setting = lambda name: re.search(rf"pub const {name}: f32 = ([0-9.]+);", main_rs).group(1)
    parts.append(MAIN.format(c_puct=setting("C_PUCT"), fpu=setting("FPU_REDUCTION")))
    code = "".join(parts)
    # The weights, broken into lines: a backslash at the end of a line continues the literal.
    lines = [text[i : i + args.line] for i in range(0, len(text), args.line)]
    weights = 'const NET: &str = "\\\n' + "\\\n".join(lines) + '";\n'
    paste = code + weights
    path = os.path.join(args.out, "tronzero_paste.rs")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(paste)

    units = len(paste.encode("utf-16-le")) // 2  # how an editor counts: UTF-16 code units
    print(f"{path}: {units} characters ({LIMIT - units} under the limit), "
          f"{len(paste.encode('utf-8'))} bytes as UTF-8")
    print(f"  code {len(code)} characters, weights {len(weights)} characters "
          f"({n_params} parameters; {args.bits} bits in the hidden layer, {args.small_bits} elsewhere)")
    print("  relative rounding error per tensor: "
          + ", ".join(f"{k} {v:.4f}" for k, v in stats.items()))
    assert units <= LIMIT, f"the paste is {units - LIMIT} characters over CodinGame's limit"


if __name__ == "__main__":
    main()
