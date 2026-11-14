//! The policy/value network's forward pass.
//!
//! Input: a 59x39 canvas centred on the mover's head (the egocentric view
//! slinky uses), sixteen planes, zero off the board. Three 3x3 stride-2 convs
//! (pad 1) take it to 30x20, 15x10 and 8x5; the flattened 8x5 map plus a few
//! scalars feed one hidden layer, then a 4-logit policy head and four tanh
//! value outputs, one per slot (the mover, then the other bikes in turn
//! order).
//!
//! Planes: 0 empty; 1-3 the heads of slots 1-3; 4-7 the cells each slot
//! reaches first (ties to the earlier slot, which moves sooner); 8-11 each
//! slot's closeness `(64 - min(d, 64)) / 64`; 12-15 each slot's trail.
//!
//! The board covers about a quarter of the canvas, so each layer is computed
//! only where its output can differ from the network's response to an empty
//! canvas (the "background", computed once at load).

use crate::board::{Feat, State, COL_MASK, MAXP, UNREACH};
// paste:skip-begin
use std::io::Read;
// paste:skip-end

pub const CW: usize = 59;
pub const CH: usize = 39;
pub const NPLANES: usize = 16;
pub const NSCAL: usize = 13;
pub const NVAL: usize = MAXP;
/// Sixteen `f32`s on a cache line.
#[repr(align(64))]
#[derive(Clone, Copy)]
struct Line([f32; 16]);

/// `f32`s whose storage starts on a 64-byte boundary. The allocator promises
/// only 16, and the AVX kernels read 32 bytes at a time: from an unaligned
/// start every other read straddles two cache lines, and whether a buffer
/// lands aligned changes from one build (or one set of weights) to the next.
#[derive(Default)]
pub struct Buf {
    lines: Vec<Line>,
    len: usize,
}

impl Buf {
    fn zeros(len: usize) -> Buf {
        Buf { lines: vec![Line([0.0; 16]); (len + 15) / 16], len }
    }

    fn from(v: &[f32]) -> Buf {
        let mut b = Buf::zeros(v.len());
        b.copy_from_slice(v);
        b
    }
}

impl std::ops::Deref for Buf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // A `Line` is sixteen `f32`s and nothing else.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr() as *const f32, self.len) }
    }
}

impl std::ops::DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr() as *mut f32, self.len) }
    }
}

/// Input width, height of each conv layer, then of the last layer's output.
const DIM_W: [usize; 4] = [59, 30, 15, 8];
const DIM_H: [usize; 4] = [39, 20, 10, 5];

pub struct Net {
    pub ch: [usize; 4],
    pub hid: usize,
    conv_w: [Buf; 3],
    conv_b: [Vec<f32>; 3],
    dense_w: Buf,
    feat_w: Buf,
    pol_w: Vec<f32>,
    pol_b: [f32; 4],
    val_w: Vec<f32>,
    val_b: [f32; NVAL],
    bg: [Buf; 3],
    hid_bg: Buf,
    /// The CPU has AVX2 and FMA; otherwise the plain loops are used.
    simd: bool,
}

/// The weights as they come out of a file or a packed string.
pub struct Parts {
    pub ch: [usize; 3],
    pub hid: usize,
    pub conv_w: [Vec<f32>; 3],
    pub conv_b: [Vec<f32>; 3],
    pub dense_w: Vec<f32>,
    pub dense_b: Vec<f32>,
    pub feat_w: Vec<f32>,
    pub pol_w: Vec<f32>,
    pub pol_b: Vec<f32>,
    pub val_w: Vec<f32>,
    pub val_b: Vec<f32>,
}

type Bbox = (usize, usize, usize, usize); // x0, x1, y0, y1, inclusive

/// Per-thread buffers of the forward pass.
pub struct Work {
    canvas: Buf,
    act: [Buf; 3],
    last_rect: Option<(usize, usize)>,
    last_bbox: [Option<Bbox>; 3],
    hid: Buf,
}

use std::arch::x86_64::*;

/// 3x3 stride-2 pad-1 convolution and ReLU on the outputs inside `bb`, for
/// `NV * 8` output channels.
///
/// Each output keeps `K` sets of accumulators in registers and spreads the
/// inputs over them, so the FMAs are not one latency chain per vector. Inputs
/// are not tested for zero: a mispredicted branch costs more than the FMAs.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_nv<const NV: usize, const K: usize>(
    inp: &[f32],
    iw: usize,
    ih: usize,
    cin: usize,
    out: &mut [f32],
    ow: usize,
    w: &[f32],
    b: &[f32],
    bb: Bbox,
) {
    let cout = NV * 8;
    let (x0, x1, y0, y1) = bb;
    let zero = _mm256_setzero_ps();
    let cin_k = cin / K * K;
    for oy in y0..=y1 {
        for ox in x0..=x1 {
            let mut acc = [[zero; NV]; K];
            for v in 0..NV {
                acc[0][v] = _mm256_loadu_ps(b.as_ptr().add(v * 8));
            }
            for ky in 0..3 {
                let iy = 2 * oy + ky;
                if iy < 1 || iy > ih {
                    continue;
                }
                let iy = iy - 1;
                for kx in 0..3 {
                    let ix = 2 * ox + kx;
                    if ix < 1 || ix > iw {
                        continue;
                    }
                    let ix = ix - 1;
                    let ip = inp.as_ptr().add((iy * iw + ix) * cin);
                    let wp = w.as_ptr().add((ky * 3 + kx) * cin * cout);
                    let mut ci = 0;
                    while ci < cin_k {
                        for k in 0..K {
                            let xv = _mm256_broadcast_ss(&*ip.add(ci + k));
                            let wr = wp.add((ci + k) * cout);
                            for v in 0..NV {
                                acc[k][v] =
                                    _mm256_fmadd_ps(xv, _mm256_loadu_ps(wr.add(v * 8)), acc[k][v]);
                            }
                        }
                        ci += K;
                    }
                    // Leftover inputs all go to set 0: a variable set index would
                    // push the accumulators out of registers.
                    while ci < cin {
                        let xv = _mm256_broadcast_ss(&*ip.add(ci));
                        let wr = wp.add(ci * cout);
                        for v in 0..NV {
                            acc[0][v] = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wr.add(v * 8)), acc[0][v]);
                        }
                        ci += 1;
                    }
                }
            }
            let op = out.as_mut_ptr().add((oy * ow + ox) * cout);
            for v in 0..NV {
                let mut sum = acc[0][v];
                for k in 1..K {
                    sum = _mm256_add_ps(sum, acc[k][v]);
                }
                _mm256_storeu_ps(op.add(v * 8), _mm256_max_ps(sum, zero));
            }
        }
    }
}

/// `conv_nv` reading only the input planes listed in `planes` (the others
/// are known to be zero inside `bb`). The first layer uses it: a position
/// with two or three bikes leaves the planes of the unused slots empty.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_planes<const NV: usize, const K: usize>(
    inp: &[f32],
    iw: usize,
    ih: usize,
    cin: usize,
    out: &mut [f32],
    ow: usize,
    w: &[f32],
    b: &[f32],
    bb: Bbox,
    planes: &[usize],
) {
    let cout = NV * 8;
    let (x0, x1, y0, y1) = bb;
    let zero = _mm256_setzero_ps();
    let n = planes.len();
    let n_k = n / K * K;
    for oy in y0..=y1 {
        for ox in x0..=x1 {
            let mut acc = [[zero; NV]; K];
            for v in 0..NV {
                acc[0][v] = _mm256_loadu_ps(b.as_ptr().add(v * 8));
            }
            for ky in 0..3 {
                let iy = 2 * oy + ky;
                if iy < 1 || iy > ih {
                    continue;
                }
                let iy = iy - 1;
                for kx in 0..3 {
                    let ix = 2 * ox + kx;
                    if ix < 1 || ix > iw {
                        continue;
                    }
                    let ix = ix - 1;
                    let ip = inp.as_ptr().add((iy * iw + ix) * cin);
                    let wp = w.as_ptr().add((ky * 3 + kx) * cin * cout);
                    let mut j = 0;
                    while j < n_k {
                        for k in 0..K {
                            let ci = *planes.get_unchecked(j + k);
                            let xv = _mm256_broadcast_ss(&*ip.add(ci));
                            let wr = wp.add(ci * cout);
                            for v in 0..NV {
                                acc[k][v] =
                                    _mm256_fmadd_ps(xv, _mm256_loadu_ps(wr.add(v * 8)), acc[k][v]);
                            }
                        }
                        j += K;
                    }
                    while j < n {
                        let ci = *planes.get_unchecked(j);
                        let xv = _mm256_broadcast_ss(&*ip.add(ci));
                        let wr = wp.add(ci * cout);
                        for v in 0..NV {
                            acc[0][v] = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wr.add(v * 8)), acc[0][v]);
                        }
                        j += 1;
                    }
                }
            }
            let op = out.as_mut_ptr().add((oy * ow + ox) * cout);
            for v in 0..NV {
                let mut sum = acc[0][v];
                for k in 1..K {
                    sum = _mm256_add_ps(sum, acc[k][v]);
                }
                _mm256_storeu_ps(op.add(v * 8), _mm256_max_ps(sum, zero));
            }
        }
    }
}

/// The input planes that can be non-zero with two, three and four bikes in the game.
const USED_PLANES: [&[usize]; 3] = [
    &[0, 1, 4, 5, 8, 9, 12, 13],
    &[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];

/// The convolution in plain Rust, for CPUs without AVX2 and FMA.
#[allow(clippy::too_many_arguments)]
fn conv_scalar(
    inp: &[f32],
    iw: usize,
    ih: usize,
    cin: usize,
    out: &mut [f32],
    ow: usize,
    cout: usize,
    w: &[f32],
    b: &[f32],
    bb: Bbox,
) {
    let mut acc = [0.0f32; 64];
    for oy in bb.2..=bb.3 {
        for ox in bb.0..=bb.1 {
            acc[..cout].copy_from_slice(b);
            for ky in 0..3 {
                let iy = 2 * oy + ky;
                if iy < 1 || iy > ih {
                    continue;
                }
                for kx in 0..3 {
                    let ix = 2 * ox + kx;
                    if ix < 1 || ix > iw {
                        continue;
                    }
                    let i0 = ((iy - 1) * iw + ix - 1) * cin;
                    for ci in 0..cin {
                        let x = inp[i0 + ci];
                        if x != 0.0 {
                            let w0 = ((ky * 3 + kx) * cin + ci) * cout;
                            saxpy(&mut acc[..cout], x, &w[w0..w0 + cout]);
                        }
                    }
                }
            }
            let o0 = (oy * ow + ox) * cout;
            for o in 0..cout {
                out[o0 + o] = acc[o].max(0.0);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_bbox(
    simd: bool,
    inp: &[f32],
    iw: usize,
    ih: usize,
    cin: usize,
    out: &mut [f32],
    ow: usize,
    cout: usize,
    w: &[f32],
    b: &[f32],
    bb: Bbox,
) {
    assert!(inp.len() >= iw * ih * cin && w.len() == 9 * cin * cout && b.len() == cout);
    assert!(out.len() >= (bb.3 * ow + bb.1 + 1) * cout);
    if !simd {
        return conv_scalar(inp, iw, ih, cin, out, ow, cout, w, b, bb);
    }
    unsafe {
        match cout / 8 {
            1 => conv_nv::<1, 4>(inp, iw, ih, cin, out, ow, w, b, bb),
            2 => conv_nv::<2, 4>(inp, iw, ih, cin, out, ow, w, b, bb),
            3 => conv_nv::<3, 4>(inp, iw, ih, cin, out, ow, w, b, bb),
            4 => conv_nv::<4, 2>(inp, iw, ih, cin, out, ow, w, b, bb),
            6 => conv_nv::<6, 2>(inp, iw, ih, cin, out, ow, w, b, bb),
            8 => conv_nv::<8, 1>(inp, iw, ih, cin, out, ow, w, b, bb),
            _ => panic!("unsupported conv width {}", cout),
        }
    }
}

/// `hid += sum over the cells in bb of (act - bg) * dense_w`, for `NV * 8` hidden units.
#[target_feature(enable = "avx2,fma")]
unsafe fn dense_nv<const NV: usize>(
    hid: &mut [f32],
    act: &[f32],
    bg: &[f32],
    w: &[f32],
    c3: usize,
    row_w: usize,
    bb: Bbox,
) {
    let n = NV * 8;
    let mut acc = [_mm256_setzero_ps(); NV];
    for v in 0..NV {
        acc[v] = _mm256_loadu_ps(hid.as_ptr().add(v * 8));
    }
    let (x0, x1, y0, y1) = bb;
    for y in y0..=y1 {
        for i in (y * row_w + x0) * c3..(y * row_w + x1 + 1) * c3 {
            let xv = _mm256_set1_ps(*act.get_unchecked(i) - *bg.get_unchecked(i));
            let wr = w.as_ptr().add(i * n);
            for v in 0..NV {
                acc[v] = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wr.add(v * 8)), acc[v]);
            }
        }
    }
    for v in 0..NV {
        _mm256_storeu_ps(hid.as_mut_ptr().add(v * 8), acc[v]);
    }
}

#[allow(clippy::too_many_arguments)]
fn dense_bbox(simd: bool, hid: &mut [f32], act: &[f32], bg: &[f32], w: &[f32], c3: usize, row_w: usize, bb: Bbox) {
    let n = hid.len();
    assert!(act.len() == bg.len() && w.len() == act.len() * n);
    assert!((bb.3 * row_w + bb.1 + 1) * c3 <= act.len());
    if !simd {
        for y in bb.2..=bb.3 {
            for i in (y * row_w + bb.0) * c3..(y * row_w + bb.1 + 1) * c3 {
                let x = act[i] - bg[i];
                if x != 0.0 {
                    saxpy(hid, x, &w[i * n..(i + 1) * n]);
                }
            }
        }
        return;
    }
    unsafe {
        match n / 8 {
            4 => dense_nv::<4>(hid, act, bg, w, c3, row_w, bb),
            8 => dense_nv::<8>(hid, act, bg, w, c3, row_w, bb),
            12 => dense_nv::<12>(hid, act, bg, w, c3, row_w, bb),
            16 => dense_nv::<16>(hid, act, bg, w, c3, row_w, bb),
            24 => dense_nv::<24>(hid, act, bg, w, c3, row_w, bb),
            32 => dense_nv::<32>(hid, act, bg, w, c3, row_w, bb),
            _ => panic!("unsupported hidden width {}", n),
        }
    }
}

#[inline(always)]
fn saxpy(acc: &mut [f32], x: f32, w: &[f32]) {
    for (a, &wv) in acc.iter_mut().zip(w) {
        *a += x * wv;
    }
}

/// Outputs of a stride-2 pad-1 3x3 conv that see inputs in `bb`.
fn out_bbox(bb: Bbox, ow: usize, oh: usize) -> Bbox {
    (bb.0 / 2, ((bb.1 + 1) / 2).min(ow - 1), bb.2 / 2, ((bb.3 + 1) / 2).min(oh - 1))
}

// paste:skip-begin
fn read_f32s(r: &mut impl Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
// paste:skip-end

/// Packed weights are text: every character carries `CHAR_BITS` bits, as the
/// code point `CHAR_BASE + value` (all inside the CJK Unified Ideographs
/// block, so each counts as one character wherever length is counted).
pub const CHAR_BASE: u32 = 0x4E00;
pub const CHAR_BITS: u32 = 14;

/// A bit reader over packed text, most significant bit first.
struct Bits<'a> {
    chars: std::str::Chars<'a>,
    acc: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn get(&mut self, k: u32) -> u32 {
        while self.n < k {
            let c = self.chars.next().expect("packed weights are truncated") as u32;
            assert!(c >= CHAR_BASE && c < CHAR_BASE + (1 << CHAR_BITS), "bad character in packed weights");
            self.acc = (self.acc << CHAR_BITS | (c - CHAR_BASE) as u64) & ((1 << 48) - 1);
            self.n += CHAR_BITS;
        }
        self.n -= k;
        (self.acc >> self.n & ((1u64 << k) - 1)) as u32
    }

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32::from_bits(self.get(32))).collect()
    }

    /// `n` weights in `groups` groups: the bits per weight, one scale per
    /// group, then the weights. `inner`: the group is the last index
    /// (`i % groups`); otherwise the first.
    fn weights(&mut self, n: usize, groups: usize, inner: bool) -> Vec<f32> {
        let bits = self.get(8);
        let scale = self.f32s(groups);
        let zero = 1i32 << (bits - 1);
        (0..n)
            .map(|i| {
                let g = if inner { i % groups } else { i / (n / groups) };
                (self.get(bits) as i32 - zero) as f32 * scale[g]
            })
            .collect()
    }
}

impl Net {
    /// Finish a network: detect the CPU and compute the background, every
    /// layer's output on an all-zero canvas.
    pub fn from_parts(p: Parts) -> Net {
        let ch = [NPLANES, p.ch[0], p.ch[1], p.ch[2]];
        let hid = p.hid;
        assert!(ch[1..].iter().all(|&c| c % 8 == 0 && c <= 64) && p.dense_b.len() == hid);
        let simd = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && std::env::var_os("TZ_SCALAR").is_none();
        let conv_w = [Buf::from(&p.conv_w[0]), Buf::from(&p.conv_w[1]), Buf::from(&p.conv_w[2])];
        let dense_w = Buf::from(&p.dense_w);
        let mut bg: [Buf; 3] = Default::default();
        let mut prev = Buf::zeros(CW * CH * NPLANES);
        for l in 0..3 {
            let (ow, oh) = (DIM_W[l + 1], DIM_H[l + 1]);
            let mut out = Buf::zeros(ow * oh * ch[l + 1]);
            conv_bbox(
                simd, &prev, DIM_W[l], DIM_H[l], ch[l], &mut out, ow, ch[l + 1], &conv_w[l], &p.conv_b[l],
                (0, ow - 1, 0, oh - 1),
            );
            bg[l] = Buf::from(&out);
            prev = out;
        }
        let mut hid_bg = Buf::from(&p.dense_b);
        for (i, &x) in bg[2].iter().enumerate() {
            if x != 0.0 {
                saxpy(&mut hid_bg, x, &dense_w[i * hid..(i + 1) * hid]);
            }
        }
        Net {
            ch,
            hid,
            conv_w,
            conv_b: p.conv_b,
            dense_w,
            feat_w: Buf::from(&p.feat_w),
            pol_w: p.pol_w,
            pol_b: [p.pol_b[0], p.pol_b[1], p.pol_b[2], p.pol_b[3]],
            val_w: p.val_w,
            val_b: [p.val_b[0], p.val_b[1], p.val_b[2], p.val_b[3]],
            bg,
            hid_bg,
            simd,
        }
    }

    /// Read weights packed by `tools/make_paste.py`: each tensor rounded to a
    /// few bits per weight, with one scale per output channel, written as text.
    pub fn from_packed(text: &str) -> Net {
        let mut r = Bits { chars: text.chars(), acc: 0, n: 0 };
        let c = [r.get(8) as usize, r.get(8) as usize, r.get(8) as usize];
        let hid = r.get(8) as usize;
        let cin = [NPLANES, c[0], c[1]];
        let mut conv_w: [Vec<f32>; 3] = Default::default();
        let mut conv_b: [Vec<f32>; 3] = Default::default();
        for l in 0..3 {
            conv_w[l] = r.weights(9 * cin[l] * c[l], c[l], true);
            conv_b[l] = r.f32s(c[l]);
        }
        let flat = DIM_W[3] * DIM_H[3] * c[2];
        let dense_w = r.weights(flat * hid, hid, true);
        let dense_b = r.f32s(hid);
        let feat_w = r.weights(NSCAL * hid, hid, true);
        let pol_w = r.weights(4 * hid, 4, false);
        let pol_b = r.f32s(4);
        let val_w = r.weights(NVAL * hid, NVAL, false);
        let val_b = r.f32s(NVAL);
        Net::from_parts(Parts { ch: c, hid, conv_w, conv_b, dense_w, dense_b, feat_w, pol_w, pol_b, val_w, val_b })
    }

    // paste:skip-begin
    /// Read a weight file written by `train/tz.py::export_net`.
    pub fn load(path: &str) -> std::io::Result<Net> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 32];
        f.read_exact(&mut head)?;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if &head[0..4] != b"TZN2" {
            return Err(bad("not a TZN2 weight file"));
        }
        let u = |i: usize| u32::from_le_bytes([head[i], head[i + 1], head[i + 2], head[i + 3]]) as usize;
        if u(4) != NPLANES || u(8) != NSCAL || u(28) != NVAL {
            return Err(bad("plane, scalar or value count mismatch"));
        }
        let ch = [u(12), u(16), u(20)];
        let hid = u(24);
        let cin = [NPLANES, ch[0], ch[1]];
        let mut conv_w: [Vec<f32>; 3] = Default::default();
        let mut conv_b: [Vec<f32>; 3] = Default::default();
        for l in 0..3 {
            conv_w[l] = read_f32s(&mut f, 9 * cin[l] * ch[l])?;
            conv_b[l] = read_f32s(&mut f, ch[l])?;
        }
        let flat = DIM_W[3] * DIM_H[3] * ch[2];
        let dense_w = read_f32s(&mut f, flat * hid)?;
        let dense_b = read_f32s(&mut f, hid)?;
        let feat_w = read_f32s(&mut f, NSCAL * hid)?;
        let pol_w = read_f32s(&mut f, 4 * hid)?;
        let pol_b = read_f32s(&mut f, 4)?;
        let val_w = read_f32s(&mut f, NVAL * hid)?;
        let val_b = read_f32s(&mut f, NVAL)?;
        let mut rest = Vec::new();
        f.read_to_end(&mut rest)?;
        if !rest.is_empty() {
            return Err(bad("trailing bytes in weight file"));
        }
        Ok(Net::from_parts(Parts { ch, hid, conv_w, conv_b, dense_w, dense_b, feat_w, pol_w, pol_b, val_w, val_b }))
    }
    // paste:skip-end

    pub fn work(&self) -> Work {
        Work {
            canvas: Buf::zeros(CW * CH * NPLANES),
            act: [Buf::from(&self.bg[0]), Buf::from(&self.bg[1]), Buf::from(&self.bg[2])],
            last_rect: None,
            last_bbox: [None; 3],
            hid: Buf::zeros(self.hid),
        }
    }

    /// Policy logits (all four directions, unmasked) and each slot's value in
    /// [-1, 1]: its expected placement among the bikes still in, +1 for
    /// winning and -1 for being the next one out. `f` must be `features(st)`.
    pub fn eval(&self, wk: &mut Work, st: &State, f: &Feat) -> ([f32; 4], [f32; NVAL]) {
        self.eval_parts(wk, st, f, usize::MAX)
    }

    /// `eval`, stopping after the canvas (`parts` 1) or after that many conv
    /// layers plus one (2..=4); for timing the pieces.
    pub fn eval_parts(&self, wk: &mut Work, st: &State, f: &Feat, parts: usize) -> ([f32; 4], [f32; NVAL]) {
        let ns = f.nslots;
        let me = f.slot[0] as usize;
        // The board's top-left corner on the canvas: the mover's head lands on (29, 19).
        let ox = 29 - st.hx[me] as usize;
        let oy = 19 - st.hy[me] as usize;

        if let Some((px, py)) = wk.last_rect {
            for y in 0..20 {
                let s = ((y + py) * CW + px) * NPLANES;
                wk.canvas[s..s + 30 * NPLANES].fill(0.0);
            }
        }
        wk.last_rect = Some((ox, oy));

        let mut terr = [0i32; MAXP];
        let mut reach = [0i32; MAXP];
        let mut n_empty = 0;
        for y in 0..20 {
            let row = ((y + oy) * CW + ox) * NPLANES;
            let mut e = !st.occ[y] & COL_MASK;
            while e != 0 {
                let x = e.trailing_zeros() as usize;
                e &= e - 1;
                let c = row + x * NPLANES;
                let i = y * 30 + x;
                n_empty += 1;
                wk.canvas[c] = 1.0;
                let mut best = UNREACH;
                let mut owner = 0;
                for k in 0..ns {
                    let d = f.dist[k][i];
                    if d != UNREACH {
                        reach[k] += 1;
                        wk.canvas[c + 8 + k] = (64 - d.min(64) as i32) as f32 / 64.0;
                        if d < best {
                            best = d;
                            owner = k;
                        }
                    }
                }
                if best != UNREACH {
                    terr[owner] += 1;
                    wk.canvas[c + 4 + owner] = 1.0;
                }
            }
            for k in 0..ns {
                let mut t = st.trail[f.slot[k] as usize][y];
                while t != 0 {
                    let x = t.trailing_zeros() as usize;
                    t &= t - 1;
                    wk.canvas[row + x * NPLANES + 12 + k] = 1.0;
                }
            }
        }
        for k in 1..ns {
            let p = f.slot[k] as usize;
            let c = ((st.hy[p] as usize + oy) * CW + st.hx[p] as usize + ox) * NPLANES;
            wk.canvas[c + k] = 1.0;
        }
        let mut scal = [0.0f32; NSCAL];
        let mut rival = 0;
        for k in 0..ns {
            scal[k] = terr[k] as f32 / 128.0;
            scal[4 + k] = reach[k] as f32 / 256.0;
            if k > 0 {
                rival = rival.max(terr[k]);
            }
        }
        scal[8] = n_empty as f32 / 600.0;
        scal[7 + ns] = 1.0; // 9, 10, 11: two, three, four bikes left
        scal[12] = (terr[0] - rival) as f32 / 32.0;

        // Convolutions, each on the outputs the board can influence.
        let mut bb: Bbox = (ox, ox + 29, oy, oy + 19);
        if parts == 1 {
            return ([0.0; 4], [scal[0]; NVAL]);
        }
        for l in 0..3.min(parts - 1) {
            let (ow, oh, cout) = (DIM_W[l + 1], DIM_H[l + 1], self.ch[l + 1]);
            if let Some((x0, x1, y0, y1)) = wk.last_bbox[l] {
                for y in y0..=y1 {
                    let s = (y * ow + x0) * cout;
                    let e = (y * ow + x1 + 1) * cout;
                    wk.act[l][s..e].copy_from_slice(&self.bg[l][s..e]);
                }
            }
            bb = out_bbox(bb, ow, oh);
            wk.last_bbox[l] = Some(bb);
            let (lo, hi) = wk.act.split_at_mut(l);
            if l == 0 && cout == 16 && self.simd {
                assert!(wk.canvas.len() == CW * CH * NPLANES && self.conv_w[0].len() == 9 * NPLANES * 16);
                assert!(hi[0].len() >= (bb.3 * ow + bb.1 + 1) * 16);
                unsafe {
                    conv_planes::<2, 4>(
                        &wk.canvas, CW, CH, NPLANES, &mut hi[0], ow, &self.conv_w[0], &self.conv_b[0], bb,
                        USED_PLANES[ns - 2],
                    );
                }
                continue;
            }
            let inp: &[f32] = if l == 0 { &wk.canvas } else { &lo[l - 1] };
            conv_bbox(
                self.simd, inp, DIM_W[l], DIM_H[l], self.ch[l], &mut hi[0], ow, cout, &self.conv_w[l],
                &self.conv_b[l], bb,
            );
        }

        if parts <= 4 {
            return ([0.0; 4], [scal[0]; NVAL]);
        }
        // Hidden layer: the background's pre-activation plus the cells that differ.
        let hid = self.hid;
        wk.hid.copy_from_slice(&self.hid_bg);
        dense_bbox(self.simd, &mut wk.hid, &wk.act[2], &self.bg[2], &self.dense_w, self.ch[3], DIM_W[3], bb);
        for (s, &x) in scal.iter().enumerate() {
            saxpy(&mut wk.hid, x, &self.feat_w[s * hid..(s + 1) * hid]);
        }
        let mut logits = self.pol_b;
        let mut v = self.val_b;
        for (h, x) in wk.hid.iter().enumerate() {
            let x = x.max(0.0);
            if x != 0.0 {
                for a in 0..4 {
                    logits[a] += x * self.pol_w[a * hid + h];
                }
                for k in 0..NVAL {
                    v[k] += x * self.val_w[k * hid + h];
                }
            }
        }
        (logits, v.map(f32::tanh))
    }
}
