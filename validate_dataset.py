"""
Post-collection sanity checks for the landing dataset.

Run this after convert_to_yolo.py. Everything here is derived independently of the
plugin: the runway geometry comes from runways.dat, and the camera intrinsics are
solved for, so agreement between the labels and position/pose is real evidence rather
than a tautology.

    python3 validate_dataset.py [--dir DATASET] [--limit N]

Checks, in order:
  1. reprojection   -- fit one pinhole camera to every episode, report corner error.
                       A recovered principal point near the image centre and a small
                       median error means labels, position.txt and pose.txt agree.
  2. frame pairing  -- re-run the fit pairing label[i] with position/pose[i+shift] for
                       shift in -3..3. Shift 0 must win, or the video and the ground
                       truth are misaligned by a constant number of frames.
  3. glidepath      -- a coupled ILS approach is a straight line, so height vs
                       along-track must be straight. Residuals expose a height channel
                       that is tracking terrain instead of the runway.
  4. framing        -- fraction of labels whose quad is fully inside the image.
"""

import os
import sys
import math
import json
import argparse
import statistics

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dat_utils import parse_runways_dat
from utils.general_utils import compute_bearing


IMG_W, IMG_H = 1280, 720


def great_circle(lat1, lon1, lat2, lon2):
    """Metres between two lat/lon points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def body_to_runway(yaw, pitch, roll):
    """
    Rotation from body axes to the runway frame (X along the runway, Y up, Z right).
    +yaw = nose right, +pitch = nose up, +roll = right wing down.
    """
    y, p, r = map(math.radians, (yaw, pitch, roll))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    r_yaw   = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    r_pitch = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    r_roll  = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return r_yaw @ r_pitch @ r_roll


def read_rows(path):
    return [float(v) for v in open(path).read().split()]


def load_episode(ep_dir, runways):
    """Returns (corners_in_runway_frame, [(position, rotation, observed_corners), ...])."""
    meta = open(os.path.join(ep_dir, 'meta.txt')).read().split('\n')
    airport, runway_name = meta[0].split()
    r = runways.get((airport, runway_name))
    if r is None:
        return None, []

    length = great_circle(r['r1_lat'], r['r1_long'], r['r2_lat'], r['r2_long'])
    half_w = r['runway_width'] / 2.0
    # near-left, far-left, far-right, near-right, matching get_runway_corners
    corners = np.array([
        [0.0,    0.0, -half_w],
        [length, 0.0, -half_w],
        [length, 0.0,  half_w],
        [0.0,    0.0,  half_w],
    ])

    names = sorted(os.listdir(os.path.join(ep_dir, 'labels')))
    frames = []
    for name in names:
        tokens = open(os.path.join(ep_dir, 'labels', name)).read().split()
        pos = read_rows(os.path.join(ep_dir, 'position', name))
        pose = read_rows(os.path.join(ep_dir, 'pose', name))
        pts = None
        if len(tokens) == 9:          # class + 4 corners, current format
            pts = np.array([[float(tokens[k]), float(tokens[k + 1])]
                            for k in range(1, 9, 2)]) * [IMG_W, IMG_H]
        elif len(tokens) == 85:       # class + 42-point polygon, pre-quad format
            # the old builder emitted (mis-named) right edge first, so the four corners
            # sit at 41, 21, 20, 0 -- reorder to near-left, far-left, far-right, near-right
            ring = np.array([[float(tokens[k]), float(tokens[k + 1])]
                             for k in range(1, 85, 2)])
            pts = ring[[41, 21, 20, 0]] * [IMG_W, IMG_H]
        frames.append((np.array(pos), pose, pts))
    return corners, frames


def build_samples(episodes, shift=0):
    samples = []
    for corners, frames in episodes:
        if corners is None:
            continue
        for i, (_, _, pts) in enumerate(frames):
            j = i + shift
            if pts is None or not (0 <= j < len(frames)):
                continue
            pos, pose, _ = frames[j]
            # skip anything clipped by the image edge, and anything past the threshold
            if pts.min() <= 0.0 or pts[:, 0].max() >= IMG_W or pts[:, 1].max() >= IMG_H:
                continue
            if pos[0] > -50:
                continue
            samples.append((pos, body_to_runway(*pose), corners, pts))
    return samples


def fit_camera(samples):
    """Solve for focal length and principal point; returns (params, per-corner errors)."""
    def residual(q):
        focal, cx, cy = q
        out = []
        for pos, rot, corners, observed in samples:
            cam = (corners - pos) @ rot
            fwd = np.clip(cam[:, 0], 1.0, None)
            projected = np.stack([focal * cam[:, 2] / fwd + cx,
                                  -focal * cam[:, 1] / fwd + cy], axis=1)
            out.append(projected - observed)
        return np.concatenate(out).ravel()

    solved = least_squares(residual, [980.0, IMG_W / 2, IMG_H / 2], xtol=1e-14, ftol=1e-14)
    return solved.x, np.abs(solved.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None, help='dataset root (default: config data_collection.output_dir)')
    ap.add_argument('--runways', default=None, help='runways.dat (default: config dats.runways_dat_path)')
    ap.add_argument('--limit', type=int, default=60, help='episodes to use for the camera fit')
    args = ap.parse_args()

    root, runways_path = args.dir, args.runways
    if root is None or runways_path is None:
        from omegaconf import OmegaConf
        conf = OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
        root = root or conf.data_collection.output_dir
        runways_path = runways_path or conf.dats.runways_dat_path

    runways = {(r['airport_name'], r['runway_name']): r for r in parse_runways_dat(runways_path)}
    ep_names = sorted(d for d in os.listdir(root) if d.isdigit())
    print(f"dataset: {root}\nepisodes: {len(ep_names)}\n")

    loaded = [load_episode(os.path.join(root, e), runways) for e in ep_names[:args.limit]]

    # --- 1. reprojection -------------------------------------------------------
    samples = build_samples(loaded, shift=0)
    if not samples:
        print("no usable samples; are labels 4-corner quads?")
        return
    (focal, cx, cy), errs = fit_camera(samples)
    print("1. REPROJECTION")
    print(f"   focal {focal:.1f} px -> HFOV {2 * math.degrees(math.atan(IMG_W / 2 / focal)):.1f} deg")
    print(f"   principal point ({cx:.1f}, {cy:.1f})   ideal ({IMG_W / 2:.1f}, {IMG_H / 2:.1f})")
    print(f"   corner error: median {np.median(errs):.2f} px, mean {errs.mean():.2f}, "
          f"p95 {np.percentile(errs, 95):.2f}, max {errs.max():.1f}")

    per_corner = errs.reshape(-1, 8).max(axis=1)
    ranges = np.array([abs(s[0][0]) for s in samples])
    print("   by range to threshold:")
    for lo, hi in [(0, 300), (300, 700), (700, 1200), (1200, 2000), (2000, 3500)]:
        m = (ranges >= lo) & (ranges < hi)
        if m.sum():
            print(f"     {lo:5d}-{hi:5d} m  n={m.sum():5d}  median {np.median(per_corner[m]):6.2f} px"
                  f"  p95 {np.percentile(per_corner[m], 95):6.2f} px")

    # --- 2. frame pairing ------------------------------------------------------
    print("\n2. FRAME PAIRING  (shift 0 must win)")
    best, best_shift = None, None
    for shift in (-3, -2, -1, 0, 1, 2, 3):
        s = build_samples(loaded, shift=shift)
        if not s:
            continue
        _, e = fit_camera(s)
        med = float(np.median(e))
        flag = ''
        if best is None or med < best:
            best, best_shift = med, shift
        print(f"   shift {shift:+d}: median {med:6.2f} px   (n={len(s)}){flag}")
    verdict = "OK" if best_shift == 0 else f"MISALIGNED -- apply a {best_shift:+d} frame offset in convert_to_yolo.py"
    print(f"   best shift: {best_shift:+d}  -> {verdict}")

    # --- 3. glidepath straightness --------------------------------------------
    # Only over the STABILISED segment. Further out the autopilot is still capturing the
    # localiser and glideslope, and that transient is real aircraft motion, not a bad
    # height channel -- measuring through it would mask the thing this check is for.
    STABLE_FROM, STABLE_TO = -1500.0, -400.0
    print(f"\n3. GLIDEPATH STRAIGHTNESS  (height vs along-track, {STABLE_FROM:.0f}..{STABLE_TO:.0f} m)")
    resid_max, angles, settle = [], [], []
    for e in ep_names:
        pos = np.array([read_rows(os.path.join(root, e, 'position', f))
                        for f in sorted(os.listdir(os.path.join(root, e, 'position')))])
        # where does the aircraft settle onto the centreline and stay there?
        lateral = np.abs(pos[:, 2])
        for i in range(len(pos)):
            if np.all(lateral[i:] < 10.0):
                settle.append(pos[i, 0])
                break
        m = (pos[:, 0] >= STABLE_FROM) & (pos[:, 0] <= STABLE_TO)
        if m.sum() < 30:
            continue
        slope, intercept = np.polyfit(pos[m, 0], pos[m, 1], 1)
        resid_max.append(np.abs(pos[m, 1] - (slope * pos[m, 0] + intercept)).max())
        angles.append(math.degrees(math.atan(slope)))
    if resid_max:
        resid_max = np.array(resid_max)
        print(f"   max deviation from a straight line: median {np.median(resid_max):.2f} m, "
              f"p90 {np.percentile(resid_max, 90):.2f} m, max {resid_max.max():.2f} m")
        print(f"   episodes wandering >2 m: {(resid_max > 2).sum()} of {len(resid_max)} "
              f"({100.0 * (resid_max > 2).mean():.0f}%)   <- >0 means the height channel is suspect")
        print(f"   implied glidepath: median {abs(statistics.median(angles)):.2f} deg")
    if settle:
        settle = np.array(settle)
        print(f"   autopilot settles within 10 m of centreline at: median {np.median(settle):.0f} m, "
              f"earliest {settle.min():.0f} m, latest {settle.max():.0f} m")
        print("   (everything before that is capture transient -- real motion, correctly logged)")

    # --- 4. framing ------------------------------------------------------------
    print("\n4. FRAMING  (is the whole runway in shot?)")
    total = inside = empty = 0
    for e in ep_names:
        d = os.path.join(root, e, 'labels')
        for name in sorted(os.listdir(d)):
            tokens = open(os.path.join(d, name)).read().split()
            total += 1
            if not tokens:
                empty += 1
                continue
            vals = [float(v) for v in tokens[1:]]
            if vals and min(vals) > 0.0 and max(vals) < 1.0:
                inside += 1
    print(f"   labels: {total}   fully inside frame: {inside} ({100.0 * inside / max(total, 1):.1f}%)"
          f"   empty: {empty}")
    print("   target >=98%; if it is lower, raise descent.stop_dist")


if __name__ == '__main__':
    main()
