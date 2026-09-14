"""
Re-encode an already-converted dataset's PNG frames to JPEG in place.

convert_to_yolo.py writes JPEG now. This is for datasets collected before that
change: it walks <dataset>/<episode>/images/*.png, writes the JPEG beside it, and
only then removes the PNG.

The conversion is lossy and irreversible, so each frame is verified before its
source is deleted: the JPEG is written to a temp file, read back, and checked for
the same dimensions. Anything that fails is left alone with its PNG intact.

Safe to interrupt and re-run -- a frame whose .jpg already exists is skipped.

    python3 reencode_png_to_jpeg.py                     # dataset from config.yaml
    python3 reencode_png_to_jpeg.py --dir /path/to/set
    python3 reencode_png_to_jpeg.py --quality 95 --jobs 8
    python3 reencode_png_to_jpeg.py --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import cv2


def reencode(task):
    """Returns (png_bytes, jpg_bytes, status)."""
    png_path, quality, dry_run = task
    png_path = Path(png_path)
    jpg_path = png_path.with_suffix('.jpg')

    if jpg_path.exists():
        return (0, 0, 'skipped')

    try:
        png_bytes = png_path.stat().st_size
        img = cv2.imread(str(png_path))
        if img is None:
            return (0, 0, 'unreadable')

        # encode to memory: imencode takes the format from the '.jpg' argument, so it
        # does not care what the eventual filename looks like
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return (0, 0, 'encode-failed')

        # verify the encoded bytes decode back to the same image before we destroy
        # the only copy
        check = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if check is None or check.shape != img.shape:
            return (0, 0, 'verify-failed')

        if dry_run:
            return (png_bytes, len(buf), 'ok')

        # land it via a temp name so an interrupted run cannot leave a truncated
        # .jpg behind. The suffix keeps it out of any images/*.jpg glob.
        tmp_path = png_path.with_suffix('.jpg.part')
        with open(tmp_path, 'wb') as fp:
            fp.write(buf.tobytes())
        os.replace(tmp_path, jpg_path)
        png_path.unlink()
        return (png_bytes, jpg_path.stat().st_size, 'ok')

    except Exception as e:
        print(f"  {png_path}: {e}", file=sys.stderr)
        return (0, 0, 'error')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None, help='dataset root (default: config data_collection.output_dir)')
    ap.add_argument('--quality', type=int, default=None, help='JPEG quality (default: config jpeg_quality)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count(), help='worker processes')
    ap.add_argument('--dry-run', action='store_true', help='measure only, delete nothing')
    args = ap.parse_args()

    root, quality = args.dir, args.quality
    if root is None or quality is None:
        from omegaconf import OmegaConf
        conf = OmegaConf.load(Path(__file__).resolve().parent / 'config.yaml')
        root = root or conf.data_collection.output_dir
        quality = quality if quality is not None else conf.data_collection.jpeg_quality

    pngs = sorted(str(p) for p in Path(root).glob('*/images/*.png'))
    if not pngs:
        print(f"no PNG frames under {root} -- nothing to do")
        return

    print(f"dataset : {root}")
    print(f"frames  : {len(pngs)} PNG")
    print(f"quality : {quality}{'   (DRY RUN, nothing will be deleted)' if args.dry_run else ''}")
    print(f"workers : {args.jobs}\n")

    tasks = [(p, quality, args.dry_run) for p in pngs]
    png_total = jpg_total = 0
    counts = {}
    done = 0

    with Pool(args.jobs) as pool:
        for pb, jb, status in pool.imap_unordered(reencode, tasks, chunksize=16):
            png_total += pb
            jpg_total += jb
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 500 == 0 or done == len(tasks):
                pct = 100.0 * done / len(tasks)
                saved = (png_total - jpg_total) / 1073741824
                print(f"  {done}/{len(tasks)} ({pct:5.1f}%)   freed so far: {saved:6.2f} GB", flush=True)

    print("\nresult:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if png_total:
        print(f"  PNG in  : {png_total / 1073741824:.2f} GB")
        print(f"  JPEG out: {jpg_total / 1073741824:.2f} GB")
        print(f"  freed   : {(png_total - jpg_total) / 1073741824:.2f} GB "
              f"({png_total / jpg_total:.1f}x smaller)")

    failed = sum(v for k, v in counts.items() if k not in ('ok', 'skipped'))
    if failed:
        print(f"\n{failed} frame(s) failed and kept their PNG. Re-run to retry.")
        sys.exit(1)

    if not args.dry_run and any(Path(root).glob('*/**/split_*.txt')):
        print("\nNOTE: split files reference .png paths; regenerate them.")


if __name__ == '__main__':
    main()
