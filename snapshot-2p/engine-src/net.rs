//! The policy/value network's forward pass.
//!
//! Input: a 59x39 canvas centred on the mover's head (the egocentric view
//! slinky uses), six planes, zero off the board. Three 3x3 stride-2 convs
//! (pad 1) take it to 30x20, 15x10 and 8x5; the flattened 8x5 map plus a few
//! scalars feed one hidden layer, then a 4-logit policy head and a tanh value
//! head.
//!
//! The board covers about a quarter of the canvas, so each layer is computed
//! only where its output can differ from the network's response to an empty
//! canvas (the "background", computed once at load).

use crate::board::{Feat, State, CELLS, UNREACH};
use std::io::Read;

pub const CW: usize = 59;
pub const CH: usize = 39;
pub const NPLANES: usize = 6;
pub const NSCAL: usize = 6;
/// Input width, height of each conv layer, then of the last layer's output.
const DIM_W: [usize; 4] = [59, 30, 15, 8];
const DIM_H: [usize; 4] = [39, 20, 10, 5];

pub struct Net {
    pub ch: [usize; 4],
    pub hid: usize,
    conv_w: [Vec<f32>; 3],
    conv_b: [Vec<f32>; 3],
    dense_w: Vec<f32>,
    feat_w: Vec<f32>,
    pol_w: Vec<f32>,
    pol_b: [f32; 4],
    val_w: Vec<f32>,
    val_b: f32,
    bg: [Vec<f32>; 3],
    hid_bg: Vec<f32>,
}

type Bbox = (usize, usize, usize, usize); // x0, x1, y0, y1, inclusive

/// Per-thread buffers of the forward pass.
pub struct Work {
    canvas: Vec<f32>,
    act: [Vec<f32>; 3],
    last_rect: Option<(usize, usize)>,
    last_bbox: [Option<Bbox>; 3],
    hid: Vec<f32>,
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

#[allow(clippy::too_many_arguments)]
fn conv_bbox(
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
    unsafe {
        match cout / 8 {
            1 => conv_nv::<1, 4>(inp, iw, ih, cin, out, ow, w, b, bb),
            // The first layer: one accumulator set per input plane.
            2 if cin == 6 => conv_nv::<2, 6>(inp, iw, ih, cin, out, ow, w, b, bb),
            2 => conv_nv::<2, 4>(inp, iw, ih, cin, out, ow, w, b, bb),
            3 => conv_nv::<3, 4>(inp, iw, ih, cin, out, ow, w, b, bb),
            4 => conv_nv::<4, 2>(inp, iw, ih, cin, out, ow, w, b, bb),
            6 => conv_nv::<6, 2>(inp, iw, ih, cin, out, ow, w, b, bb),
            8 => conv_nv::<8, 1>(inp, iw, ih, cin, out, ow, w, b, bb),
            _ => panic!("unsupported conv width {cout}"),
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

fn dense_bbox(hid: &mut [f32], act: &[f32], bg: &[f32], w: &[f32], c3: usize, row_w: usize, bb: Bbox) {
    let n = hid.len();
    assert!(act.len() == bg.len() && w.len() == act.len() * n);
    assert!((bb.3 * row_w + bb.1 + 1) * c3 <= act.len());
    unsafe {
        match n / 8 {
            4 => dense_nv::<4>(hid, act, bg, w, c3, row_w, bb),
            8 => dense_nv::<8>(hid, act, bg, w, c3, row_w, bb),
            12 => dense_nv::<12>(hid, act, bg, w, c3, row_w, bb),
            16 => dense_nv::<16>(hid, act, bg, w, c3, row_w, bb),
            24 => dense_nv::<24>(hid, act, bg, w, c3, row_w, bb),
            32 => dense_nv::<32>(hid, act, bg, w, c3, row_w, bb),
            _ => panic!("unsupported hidden width {n}"),
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

fn read_f32s(r: &mut impl Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

impl Net {
    /// Read a weight file written by `train/tz.py::export_net`.
    pub fn load(path: &str) -> std::io::Result<Net> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 28];
        f.read_exact(&mut head)?;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if &head[0..4] != b"TZN1" {
            return Err(bad("not a TZN1 weight file"));
        }
        let u = |i: usize| u32::from_le_bytes([head[i], head[i + 1], head[i + 2], head[i + 3]]) as usize;
        if u(4) != NPLANES || u(8) != NSCAL {
            return Err(bad("plane or scalar count mismatch"));
        }
        let ch = [NPLANES, u(12), u(16), u(20)];
        let hid = u(24);
        let mut conv_w: [Vec<f32>; 3] = Default::default();
        let mut conv_b: [Vec<f32>; 3] = Default::default();
        for l in 0..3 {
            conv_w[l] = read_f32s(&mut f, 9 * ch[l] * ch[l + 1])?;
            conv_b[l] = read_f32s(&mut f, ch[l + 1])?;
        }
        let flat = DIM_W[3] * DIM_H[3] * ch[3];
        let dense_w = read_f32s(&mut f, flat * hid)?;
        let dense_b = read_f32s(&mut f, hid)?;
        let feat_w = read_f32s(&mut f, NSCAL * hid)?;
        let pol_w = read_f32s(&mut f, 4 * hid)?;
        let pb = read_f32s(&mut f, 4)?;
        let val_w = read_f32s(&mut f, hid)?;
        let val_b = read_f32s(&mut f, 1)?[0];
        let mut rest = Vec::new();
        f.read_to_end(&mut rest)?;
        if !rest.is_empty() {
            return Err(bad("trailing bytes in weight file"));
        }

        // The background: every layer's output on an all-zero canvas.
        let mut bg: [Vec<f32>; 3] = Default::default();
        let mut prev = vec![0.0f32; CW * CH * NPLANES];
        for l in 0..3 {
            let (ow, oh) = (DIM_W[l + 1], DIM_H[l + 1]);
            let mut out = vec![0.0f32; ow * oh * ch[l + 1]];
            conv_bbox(
                &prev, DIM_W[l], DIM_H[l], ch[l], &mut out, ow, ch[l + 1], &conv_w[l], &conv_b[l],
                (0, ow - 1, 0, oh - 1),
            );
            bg[l] = out.clone();
            prev = out;
        }
        let mut hid_bg = dense_b;
        for (i, &x) in bg[2].iter().enumerate() {
            if x != 0.0 {
                saxpy(&mut hid_bg, x, &dense_w[i * hid..(i + 1) * hid]);
            }
        }
        Ok(Net {
            ch,
            hid,
            conv_w,
            conv_b,
            dense_w,
            feat_w,
            pol_w,
            pol_b: [pb[0], pb[1], pb[2], pb[3]],
            val_w,
            val_b,
            bg,
            hid_bg,
        })
    }

    pub fn work(&self) -> Work {
        Work {
            canvas: vec![0.0; CW * CH * NPLANES],
            act: [self.bg[0].clone(), self.bg[1].clone(), self.bg[2].clone()],
            last_rect: None,
            last_bbox: [None; 3],
            hid: vec![0.0; self.hid],
        }
    }

    /// Policy logits (all four directions, unmasked) and the value in [-1, 1]
    /// for the player to move. `f` must be `features(st)`.
    pub fn eval(&self, wk: &mut Work, st: &State, f: &Feat) -> ([f32; 4], f32) {
        self.eval_parts(wk, st, f, usize::MAX)
    }

    /// `eval`, stopping after the canvas (`parts` 1) or after that many conv
    /// layers plus one (2..=4); for timing the pieces.
    pub fn eval_parts(&self, wk: &mut Work, st: &State, f: &Feat, parts: usize) -> ([f32; 4], f32) {
        let me = st.to_move as usize;
        let opp = me ^ 1;
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

        let (mut terr_me, mut terr_opp, mut reach_me, mut reach_opp, mut n_empty) = (0, 0, 0, 0, 0);
        for i in 0..CELLS {
            let (x, y) = (i % 30, i / 30);
            let c = ((y + oy) * CW + x + ox) * NPLANES;
            if st.occ[y] >> x & 1 != 0 {
                continue; // the rect was cleared; occupied cells stay all-zero
            }
            n_empty += 1;
            let (dm, dop) = (f.dist[0][i], f.dist[1][i]);
            wk.canvas[c] = 1.0;
            if dm != UNREACH {
                reach_me += 1;
                wk.canvas[c + 4] = (64 - dm.min(64) as i32) as f32 / 64.0;
                if dm <= dop {
                    terr_me += 1;
                    wk.canvas[c + 2] = 1.0;
                }
            }
            if dop != UNREACH {
                reach_opp += 1;
                wk.canvas[c + 5] = (64 - dop.min(64) as i32) as f32 / 64.0;
                if dop < dm {
                    terr_opp += 1;
                    wk.canvas[c + 3] = 1.0;
                }
            }
        }
        let oc = ((st.hy[opp] as usize + oy) * CW + st.hx[opp] as usize + ox) * NPLANES;
        wk.canvas[oc + 1] = 1.0;
        let scal: [f32; NSCAL] = [
            terr_me as f32 / 128.0,
            terr_opp as f32 / 128.0,
            (terr_me - terr_opp) as f32 / 32.0,
            reach_me as f32 / 256.0,
            reach_opp as f32 / 256.0,
            n_empty as f32 / 600.0,
        ];

        // Convolutions, each on the outputs the board can influence.
        let mut bb: Bbox = (ox, ox + 29, oy, oy + 19);
        if parts == 1 {
            return ([0.0; 4], scal[0]);
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
            let inp: &[f32] = if l == 0 { &wk.canvas } else { &lo[l - 1] };
            conv_bbox(
                inp, DIM_W[l], DIM_H[l], self.ch[l], &mut hi[0], ow, cout, &self.conv_w[l],
                &self.conv_b[l], bb,
            );
        }

        if parts <= 4 {
            return ([0.0; 4], scal[0]);
        }
        // Hidden layer: the background's pre-activation plus the cells that differ.
        let hid = self.hid;
        wk.hid.copy_from_slice(&self.hid_bg);
        dense_bbox(&mut wk.hid, &wk.act[2], &self.bg[2], &self.dense_w, self.ch[3], DIM_W[3], bb);
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
                v += x * self.val_w[h];
            }
        }
        (logits, v.tanh())
    }
}
