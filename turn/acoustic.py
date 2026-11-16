"""Levels 1-3 of the hierarchy: how something SOUNDS, not what it says.

Level 1  frame features (40 ms window, 20 ms hop): loudness, pitch F0 (YIN), periodicity (how "voiced/clean" the
         sound is), spectral centroid / tilt / high-frequency share (clarity), spectral flux.
Level 2  source cues per segment: near vs far (high frequencies lost with distance; room reverberation makes the
         loudness envelope decay slowly and flattens its dynamics), periodic voice vs noise.
Level 3  paralinguistics per segment: arousal (pitch higher and wider, louder, faster than the caller's own
         baseline), final intonation (pitch slope over the last voiced 300 ms: rising vs falling), speaking rate.

Everything is RELATIVE to the caller's own first answer (U0), so it adapts to any voice, microphone and line.
Pure numpy: about 1 ms per second of audio.
"""
from __future__ import annotations

import numpy as np

SR = 16000
WIN, HOP = 640, 320
FMIN, FMAX = 70.0, 420.0
NFFT = 1024
_FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
_HANN = np.hanning(WIN)


def frames(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) / 32768.0
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def yin(fr: np.ndarray, thresh: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """F0 (Hz, NaN when unvoiced) and periodicity (1 - aperiodicity, 0..1) per frame. Vectorised YIN."""
    W = fr.shape[1]
    tmin, tmax = int(SR / FMAX), int(SR / FMIN)
    f = np.fft.rfft(fr, 2 * W, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, : tmax + 2]
    sq = np.cumsum(fr ** 2, axis=1)
    e0 = sq[:, -1:]                                      # energy of the frame
    tau = np.arange(tmax + 2)
    e_tail = e0 - np.concatenate([np.zeros((fr.shape[0], 1)), sq[:, : tmax + 1]], axis=1)  # energy of x[tau:]
    e_head = np.take(sq, W - 1 - tau, axis=1)            # energy of x[:W-tau]
    d = e_head + e_tail - 2 * ac
    d[:, 0] = 0
    cs = np.cumsum(d[:, 1:], axis=1) / np.arange(1, d.shape[1])
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] / np.maximum(cs, 1e-12)
    seg = cmnd[:, tmin:tmax]
    below = seg < thresh
    first = np.where(below.any(1), below.argmax(1), seg.argmin(1))
    # walk to the local minimum after the first dip
    best = first.copy()
    for i in range(len(best)):
        j = best[i]
        while j + 1 < seg.shape[1] and seg[i, j + 1] < seg[i, j]:
            j += 1
        best[i] = j
    ap = seg[np.arange(len(best)), best]
    t = best + tmin
    # parabolic interpolation
    i0 = np.clip(t - 1, 1, cmnd.shape[1] - 2)
    a, b, c = cmnd[np.arange(len(t)), i0], cmnd[np.arange(len(t)), i0 + 1], cmnd[np.arange(len(t)), np.minimum(i0 + 2, cmnd.shape[1] - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(a - 2 * b + c) > 1e-9, 0.5 * (a - c) / (a - 2 * b + c), 0.0)
    f0 = SR / (i0 + 1 + np.clip(off, -1, 1))
    energy = np.sqrt(e0[:, 0] / W)
    voiced = (ap < 0.35) & (energy > 10 ** (-55 / 20))
    return np.where(voiced, f0, np.nan), np.clip(1 - ap, 0, 1)


def frame_features(x: np.ndarray) -> dict:
    fr = frames(x)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    f0, per = yin(fr)
    spec = np.abs(np.fft.rfft(fr * _HANN, NFFT, axis=1)) ** 2 + 1e-12
    tot = spec.sum(1)
    centroid = (spec * _FREQS).sum(1) / tot
    lo = spec[:, (_FREQS >= 300) & (_FREQS < 1000)].sum(1)
    hi = spec[:, (_FREQS >= 2000) & (_FREQS < 7000)].sum(1)
    hf = 10 * np.log10(hi / lo)
    effort = 10 * np.log10(spec[:, (_FREQS >= 2000) & (_FREQS < 5000)].sum(1))
    lf, lp = np.log10(_FREQS[3:]), np.log10(spec[:, 3:])
    tilt = ((lf - lf.mean()) * (lp - lp.mean(1, keepdims=True))).sum(1) / ((lf - lf.mean()) ** 2).sum()
    ls = np.log(spec)
    flux = np.r_[0, np.sqrt(((np.diff(ls, axis=0)) ** 2).mean(1))]
    return {"db": db, "f0": f0, "per": per, "centroid": centroid, "hf": hf, "tilt": tilt, "flux": flux, "effort": effort}


def _st(f, ref):            # semitones relative to a reference frequency
    return 12 * np.log2(f / ref)


def segment_profile(x: np.ndarray) -> dict:
    """Summary of one segment of audio (any length >= 80 ms)."""
    F = frame_features(x)
    db = F["db"]
    act = db > db.max() - 25                               # frames that carry the sound (not the gaps)
    f0 = F["f0"]
    vf0 = f0[~np.isnan(f0)]
    # final intonation: slope of the last voiced 300 ms (semitones per second)
    slope = np.nan
    vi = np.where(~np.isnan(f0))[0]
    if len(vi) >= 5:
        tail = vi[vi >= vi[-1] - 15]
        if len(tail) >= 4:
            st = _st(f0[tail], np.median(vf0))
            slope = float(np.polyfit(tail * HOP / SR, st, 1)[0])
    # reverberation proxy: how fast loudness falls after its peaks (dB/s); far / reverberant = slow
    env = db
    decays = []
    for k in range(1, len(env) - 6):
        if env[k] >= env[k - 1] and env[k] > env[k + 1] and env[k] > env.max() - 15:
            seg = env[k:k + 6]
            decays.append(np.polyfit(np.arange(len(seg)) * HOP / SR, seg, 1)[0])
    # speaking rate: loudness peaks per second of active sound
    sm = np.convolve(env, np.ones(3) / 3, "same")
    peaks = [k for k in range(1, len(sm) - 1) if sm[k] > sm[k - 1] and sm[k] >= sm[k + 1] and sm[k] > sm.max() - 12]
    dur = len(x) / SR
    return {
        "dur": dur,
        "level": float(np.percentile(db[act], 80)) if act.any() else -90.0,
        "dyn": float(np.percentile(db[act], 90) - np.percentile(db[act], 20)) if act.sum() > 3 else 0.0,
        "voiced": float(np.mean(~np.isnan(f0[act]))) if act.any() else 0.0,
        "per": float(np.mean(F["per"][act])) if act.any() else 0.0,
        "f0": float(np.median(vf0)) if len(vf0) else np.nan,
        "f0_range": float(np.percentile(_st(vf0, np.median(vf0)), 90) - np.percentile(_st(vf0, np.median(vf0)), 10)) if len(vf0) > 4 else 0.0,
        "f0_slope": slope,
        "hf": float(np.median(F["hf"][act])) if act.any() else -40.0,
        "effort": float(np.percentile(F["effort"][act], 80)) if act.any() else -60.0,
        "tilt": float(np.median(F["tilt"][act])) if act.any() else 0.0,
        "centroid": float(np.median(F["centroid"][act])) if act.any() else 0.0,
        "flux": float(np.mean(F["flux"][act])) if act.any() else 0.0,
        "decay": float(np.median(decays)) if decays else 0.0,
        "rate": len(peaks) / max(0.2, act.sum() * HOP / SR),
    }


def relative(seg: dict, ref: dict | None) -> dict:
    """Level 2-3 cues: the segment compared with the caller's own voice (U0)."""
    if not ref:
        ref = {"level": -21.0, "dyn": 18.0, "f0": np.nan, "hf": -18.0, "tilt": -1.0, "centroid": 1500.0, "decay": -150.0,
               "rate": 4.0, "per": 0.6, "effort": -30.0}
    f0_rel = float(_st(seg["f0"], ref["f0"])) if not (np.isnan(seg["f0"]) or np.isnan(ref["f0"])) else 0.0
    out = {
        "level_rel": seg["level"] - ref["level"],
        "dyn_rel": seg["dyn"] - ref["dyn"],
        "hf_rel": seg["hf"] - ref["hf"],
        "effort_rel": seg["effort"] - ref.get("effort", -30.0),
        "tilt_rel": seg["tilt"] - ref["tilt"],
        "centroid_rel": seg["centroid"] - ref["centroid"],
        "decay_rel": seg["decay"] - ref["decay"],
        "per_rel": seg["per"] - ref["per"],
        "f0_rel": f0_rel,
        "f0_range": seg["f0_range"],
        "f0_slope": 0.0 if np.isnan(seg["f0_slope"]) else seg["f0_slope"],
        "rate_rel": seg["rate"] - ref["rate"],
        "voiced": seg["voiced"],
        "dur": seg["dur"],
    }
    # transparent composite cues (the learned layer may weigh them differently)
    out["far_score"] = -0.08 * out["hf_rel"] - 0.06 * out["dyn_rel"] + 0.004 * out["decay_rel"] - 0.05 * out["level_rel"]
    out["arousal"] = 0.08 * out["level_rel"] + 0.10 * f0_rel + 0.08 * out["f0_range"] + 0.10 * out["rate_rel"]
    return out
