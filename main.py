#!/usr/bin/env python3
"""
fencing_cuts.py — Automatic touch-by-touch cut detection
in fencing videos filmed with a panning (panoramic) camera.

Principle (cf. discussion):
  1. Ego-motion: we estimate the camera motion (panning) by
     tracking background feature points (KLT) + a robust similarity model (RANSAC).
     -> "pan_speed" signal = panning speed (coarse action indicator).
  2. Compensated motion: we register the previous frame onto the current one, then
     measure the residual = the fencers' real motion (free of the background sweep).
     -> "fg_energy" signal = foreground motion energy.
  3. Segmentation: rest = sustained low activity; touch = active segment
     between two rests. We output a plot + a CSV of the cut points.

Usage:
    python fencing_cuts.py myvideo.mp4
    python fencing_cuts.py myvideo.mp4 --out plot.png --csv cuts.csv
    python fencing_cuts.py myvideo.mp4 --high 0.30 --low 0.15 --min-rest 1.0
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_video(path, target_width=480, border=0.08, diff_thresh=20):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, prev = cap.read()
    if not ok:
        sys.exit("Empty video.")

    h0, w0 = prev.shape[:2]
    scale = target_width / float(w0)
    size = (int(round(w0 * scale)), int(round(h0 * scale)))
    prev_g = cv2.cvtColor(cv2.resize(prev, size), cv2.COLOR_BGR2GRAY)

    w, h = size
    bx, by = int(w * border), int(h * border)

    lk = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    pan_speed, fg_energy = [], []
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur_g = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(
            prev_g, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7
        )
        M = None
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None, **lk)
            if p1 is not None:
                good0, good1 = p0[st == 1], p1[st == 1]
                if len(good0) >= 8:
                    M, _ = cv2.estimateAffinePartial2D(
                        good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3
                    )

        if M is not None:
            tx, ty = M[0, 2], M[1, 2]
            pan = float(np.hypot(tx, ty))
            warped = cv2.warpAffine(prev_g, M, (w, h))
        else:
            pan = 0.0
            warped = prev_g

        a = cv2.GaussianBlur(cur_g, (5, 5), 0)
        b = cv2.GaussianBlur(warped, (5, 5), 0)
        diff = cv2.absdiff(a, b)[by : h - by, bx : w - bx]
        mask = (diff > diff_thresh).astype(np.float32)
        fg = float(mask.mean())

        pan_speed.append(pan)
        fg_energy.append(fg)

        prev_g = cur_g
        idx += 1
        if idx % 250 == 0:
            print(f"  ... {idx}/{n_total} frames", file=sys.stderr)

    cap.release()
    return np.array(pan_speed), np.array(fg_energy), fps


def smooth(x, fps, seconds=0.4):
    return gaussian_filter1d(x, sigma=max(1.0, fps * seconds / 2.0))


def normalize(x):
    """Robust normalization to ~[0, 1] (median -> 95th percentile)."""
    med = np.median(x)
    p95 = np.percentile(x, 95)
    return np.clip((x - med) / (p95 - med + 1e-6), 0, 1.5)


def segment(
    activity,
    fps,
    high=0.30,
    low=0.15,
    min_rest=0.8,
    min_touch=0.4,
    pre_pad=0.5,
    post_pad=1.0,
):
    n = len(activity)
    active = np.zeros(n, bool)
    state = False
    for i in range(n):
        if not state and activity[i] > high:
            state = True
        elif state and activity[i] < low:
            state = False
        active[i] = state

    segs = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            segs.append([i, j - 1])
            i = j
        else:
            i += 1

    min_rest_f = int(min_rest * fps)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= min_rest_f:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    min_touch_f = int(min_touch * fps)
    pre_f, post_f = int(pre_pad * fps), int(post_pad * fps)
    touches = []
    for s, e in merged:
        if e - s < min_touch_f:
            continue
        touches.append((max(0, s - pre_f), min(n - 1, e + post_f)))
    return touches


def plot(times, fg_n, pan_n, activity, touches, fps, high, low, out):
    fig, ax = plt.subplots(figsize=(14, 5.5))

    ax.plot(times, fg_n, color="#1f77b4", lw=1.4, label="Fencer motion (compensated)")
    ax.plot(
        times,
        pan_n,
        color="#aaaaaa",
        lw=1.0,
        alpha=0.8,
        label="Camera panning speed",
    )

    ax.axhline(
        high, color="#d62728", ls=":", lw=1, alpha=0.7, label=f"high threshold ({high})"
    )
    ax.axhline(
        low, color="#2ca02c", ls=":", lw=1, alpha=0.7, label=f"low threshold ({low})"
    )

    for k, (s, e) in enumerate(touches):
        ts, te = s / fps, e / fps
        ax.axvspan(ts, te, color="#ffe08a", alpha=0.35, zorder=0)
        ax.axvline(ts, color="#2ca02c", ls="--", lw=1.6)
        ax.axvline(te, color="#d62728", ls="--", lw=1.6)
        ax.text(ts, 1.46, f"#{k + 1}", color="#2ca02c", fontsize=8, ha="left", va="top")

    ax.plot([], [], color="#2ca02c", ls="--", label="Start (allez)")
    ax.plot([], [], color="#d62728", ls="--", label="End of touch")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity (normalized)")
    ax.set_title(f"Detected cut points — {len(touches)} touch(es)")
    ax.set_ylim(-0.05, 1.5)
    ax.set_xlim(times[0], times[-1])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Plot: {out}")


def cut_video(video, touches, fps, out_dir, jobs=4, crf=17, preset="slow"):
    """Cut each detected touch out of `video` into `out_dir` using ffmpeg,
    running up to `jobs` re-encodes in parallel. Returns the number of clips
    that were written successfully."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]

    def run_one(job):
        k, s, e = job
        start, dur = s / fps, (e - s) / fps
        out = os.path.join(out_dir, f"{stem}_touch_{k:02d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-i", video,
            "-t", f"{dur:.2f}",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-c:a", "aac",
            out,
        ]
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return k, out, proc.returncode, proc.stderr.decode("utf-8", "replace")

    jobs_list = [(k, s, e) for k, (s, e) in enumerate(touches, 1)]
    print(
        f"\nCutting {len(jobs_list)} touch(es) into {out_dir}/ "
        f"({jobs} parallel job(s))...",
        file=sys.stderr,
    )

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(run_one, j) for j in jobs_list]
        for fut in as_completed(futures):
            k, out, rc, err = fut.result()
            if rc == 0:
                ok += 1
                print(f"  [ok] touch {k:02d} -> {out}", file=sys.stderr)
            else:
                print(f"  [FAIL] touch {k:02d} (ffmpeg exit {rc})", file=sys.stderr)
                tail = "\n".join(err.strip().splitlines()[-3:])
                if tail:
                    print(f"         {tail}", file=sys.stderr)

    print(f"Done! {ok}/{len(jobs_list)} clip(s) written to {out_dir}/", file=sys.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Touch-by-touch cut detection (fencing)."
    )
    ap.add_argument("video")
    ap.add_argument("--out", default="cuts_plot.png", help="plot image")
    ap.add_argument("--csv", default="cuts.csv", help="CSV of cut points")
    ap.add_argument("--width", type=int, default=480, help="processing width (px)")
    ap.add_argument(
        "--high", type=float, default=0.30, help="high threshold (enter touch)"
    )
    ap.add_argument(
        "--low", type=float, default=0.15, help="low threshold (exit touch)"
    )
    ap.add_argument(
        "--min-rest", type=float, default=0.8, help="min rest between touches (s)"
    )
    ap.add_argument(
        "--min-touch", type=float, default=0.4, help="min duration of a touch (s)"
    )
    ap.add_argument(
        "--pre-pad", type=float, default=0.5, help="margin before the start (s)"
    )
    ap.add_argument(
        "--post-pad", type=float, default=1.0, help="margin after the end (s)"
    )
    ap.add_argument(
        "--pan-weight",
        type=float,
        default=0.0,
        help="weight of the panning signal in the activity (0 = ignore)",
    )
    ap.add_argument(
        "--cut",
        action="store_true",
        help="auto-cut the detected touches into clips with ffmpeg (in parallel)",
    )
    ap.add_argument(
        "--cut-dir", default="./output", help="output folder for the clips (default: ./output)"
    )
    ap.add_argument(
        "-j", "--jobs", type=int, default=4, help="number of parallel ffmpeg jobs (default: 4)"
    )
    ap.add_argument("--crf", type=int, default=17, help="x264 CRF quality, lower is better (default: 17)")
    ap.add_argument("--preset", default="slow", help="x264 encoding preset (default: slow)")
    args = ap.parse_args()

    print("Analyzing video (ego-motion + compensated motion)...", file=sys.stderr)
    pan, fg, fps = analyze_video(args.video, target_width=args.width)
    print(f"  {len(fg)} frames analyzed at {fps:.1f} fps", file=sys.stderr)

    pan_s = smooth(pan, fps)
    fg_s = smooth(fg, fps)
    fg_n = normalize(fg_s)
    pan_n = normalize(pan_s)

    activity = fg_n + args.pan_weight * pan_n
    if args.pan_weight:
        activity = np.clip(activity / (1 + args.pan_weight), 0, 1.5)

    touches = segment(
        activity,
        fps,
        high=args.high,
        low=args.low,
        min_rest=args.min_rest,
        min_touch=args.min_touch,
        pre_pad=args.pre_pad,
        post_pad=args.post_pad,
    )

    times = np.arange(len(fg)) / fps
    plot(times, fg_n, pan_n, activity, touches, fps, args.high, args.low, args.out)

    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["touch", "start_s", "end_s", "start_frame", "end_frame", "duration_s"]
        )
        for k, (s, e) in enumerate(touches, 1):
            wr.writerow(
                [k, f"{s / fps:.2f}", f"{e / fps:.2f}", s, e, f"{(e - s) / fps:.2f}"]
            )
    print(f"CSV       : {args.csv}")

    if args.cut:
        cut_video(
            args.video,
            touches,
            fps,
            args.cut_dir,
            jobs=args.jobs,
            crf=args.crf,
            preset=args.preset,
        )
    else:
        print(f"\n{len(touches)} touch(es) detected. Cut commands:")
        for k, (s, e) in enumerate(touches, 1):
            ts, dur = s / fps, (e - s) / fps
            print(
                f'  ffmpeg -ss {ts:.2f} -i "{args.video}" -t {dur:.2f} '
                f"-c copy touch_{k:02d}.mp4"
            )
        print("\nTip: pass --cut to auto-cut these into ./output/ in parallel.")


if __name__ == "__main__":
    main()
