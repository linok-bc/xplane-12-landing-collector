import os
import tqdm
import glob
from pathlib import Path
from omegaconf import OmegaConf
import cv2

# ============================================================================
# CONFIGURATION
# 
# Global variables used to control flight behavior. Read from conf_path for
# defaults. Avoid overwriting these.
# ============================================================================
conf_path = Path(__file__).resolve().parent / 'config.yaml'
conf = OmegaConf.load(conf_path)
dc =  conf.data_collection

# === DATA COLLECTION ===
OUTPUT_DIR = dc.output_dir
MAX_RUNWAYS = dc.max_runways
SIM_SPEED = dc.sim_speed
SKIP_FIRST = dc.skip_first_frames
SKIP_LAST = dc.skip_last_frames

if __name__ == '__main__':

    # (1) collect the valid directories; note that we may have removed some
    dnames = [Path(OUTPUT_DIR) / f'{idx:06}' for idx in range(1, MAX_RUNWAYS+1)]
    dnames = [dname for dname in dnames if (dname.exists() and dname.is_dir())]

    n_empty = 0
    n_clamped = 0
    n_frames = 0

    for dname in tqdm.tqdm(dnames):
        # (2) get the relevant avi, load into memory
        try:
            avi_path = sorted(glob.glob(str(dname / '*.avi')))[-1]
        except IndexError:      # file not found; we've probably already gone past
            tqdm.tqdm.write(f"Skipping {dname}, since avi not found")
            continue

        vid = cv2.VideoCapture(str(avi_path))
        frames = []
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == False:
                break
            frames.append(frame)
            
        # (3) get the relevant polygons, positions, and poses
        poly = dname / "poly.txt"
        position = dname / "position.txt"
        pose = dname / "pose.txt"

        with open(poly, 'r') as fp:
            poly = list(fp)
        with open(position, 'r') as fp:
            position = list(fp)
        with open(pose, 'r') as fp:
            pose = list(fp)

        # (4) align frames with polygons + others; align based on the last
        max_len_align = min(len(frames), len(poly), len(position), len(pose))
        frames = frames[-max_len_align:]
        poly = poly[-max_len_align:]
        position = position[-max_len_align:]
        pose = pose[-max_len_align:]

        # (5) skip relevant frames. SKIP_LAST of 0 has to become None, since a slice
        #     ending at -0 is a slice ending at 0 and would throw everything away
        end = max_len_align - SKIP_LAST if SKIP_LAST else None
        frames = frames[SKIP_FIRST:end]
        poly = poly[SKIP_FIRST:end]
        position = position[SKIP_FIRST:end]
        pose = pose[SKIP_FIRST:end]

        # (6) write frames to disk
        os.makedirs(dname / 'images', exist_ok=True)
        os.makedirs(dname / 'labels', exist_ok=True)
        os.makedirs(dname / 'position', exist_ok=True)
        os.makedirs(dname / 'pose', exist_ok=True)
        
        for idx in range(len(frames)):
            cv2.imwrite(str(dname / 'images' / f'{idx:06d}.png'), frames[idx])
            # an empty poly line means the runway was not projectable for that frame.
            # write a genuinely empty label ("no objects") rather than a bare class id,
            # which is a malformed YOLO row
            coords = poly[idx].strip()
            with open(dname / 'labels' / f'{idx:06d}.txt', 'w') as fp:
                if coords:
                    fp.write("0 " + coords + "\n")
                else:
                    n_empty += 1
            with open(dname / 'position' / f'{idx:06d}.txt', 'w') as fp:
                fp.write(position[idx].strip())
            with open(dname / 'pose' / f'{idx:06d}.txt', 'w') as fp:
                fp.write(pose[idx].strip())

        # (6b) count labels whose quad touches the image border -- with the capture
        #      cutoff in place this should stay near zero; a rising number means the
        #      cutoff is too close to the threshold for these runways
        for line in poly:
            vals = [float(v) for v in line.split()]
            n_frames += 1
            if vals and (min(vals) <= 0.0 or max(vals) >= 1.0):
                n_clamped += 1

        # (7) remove avi, ground truth txt files
        Path(avi_path).unlink()
        (dname / "poly.txt").unlink()
        (dname / "position.txt").unlink()
        (dname / "pose.txt").unlink()

    print(f"\nframes written:      {n_frames}")
    print(f"empty labels:        {n_empty} ({100.0 * n_empty / max(n_frames, 1):.2f}%)  runway not projectable")
    print(f"quads touching edge: {n_clamped} ({100.0 * n_clamped / max(n_frames, 1):.2f}%)  raise descent.stop_dist if this grows")
