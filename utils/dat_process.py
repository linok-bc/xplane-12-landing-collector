import os
import sys
import time
import math
import random
import shutil
from omegaconf import OmegaConf
from pathlib import Path
from contextlib import suppress

from dat_utils import get_runways
from general_utils import compute_bearing


conf_path = (Path(__file__).resolve() / '../../config.yaml').resolve()
conf = OmegaConf.load(conf_path)

# === DAT FILES ===
AIRPORT_DAT_PATH    = conf.dats.airport_dat_path
NAV_DAT_PATH        = conf.dats.nav_dat_path
RUNWAYS_DAT_PATH      = conf.dats.runways_dat_path

# === FILTERING ===
# The autopilot flies the ILS localiser beam, whose bearing comes from earth_nav.dat.
# The labels describe the pavement, whose bearing comes from the two runway-end
# coordinates. When those disagree the aircraft lands beside the runway: the beam is
# anchored at the localiser antenna out past the far end, so the lateral error at
# touchdown is roughly runway_length * sin(mismatch). EGVO 27 disagrees by 1.62 deg
# and put the aircraft 53.6 m off the centreline, on the grass.
MAX_BEARING_MISMATCH = 0.5   # degrees

if __name__ == '__main__':
    runways = get_runways(AIRPORT_DAT_PATH, NAV_DAT_PATH)
    kept = dropped = 0
    with open(RUNWAYS_DAT_PATH, 'w') as fp:
        for runway in runways:
            airport_name, runway_name = runway[0:2]
            elev, runway_width = runway[2]['elev'], runway[2]['runway_width']
            r1_name, r1_lat, r1_long = runway[2]['r1_name'], runway[2]['r1_lat'], runway[2]['r1_long']
            r2_name, r2_lat, r2_long = runway[2]['r2_name'], runway[2]['r2_lat'], runway[2]['r2_long']
            ils, heading = runway[2]['ils'], runway[2]['heading']

            # drop runways whose beam and pavement point different ways
            geom_heading = compute_bearing(r1_lat, r1_long, r2_lat, r2_long)
            mismatch = abs((heading - geom_heading + 180.0) % 360.0 - 180.0)
            if mismatch > MAX_BEARING_MISMATCH:
                print(f"drop {airport_name} {runway_name}: ils={heading:.3f} pavement={geom_heading:.3f} mismatch={mismatch:.2f} deg")
                dropped += 1
                continue

            kept += 1
            fp.write(f"{airport_name} {runway_name} {elev} {runway_width} {r1_name} {r1_lat} {r1_long} {r2_name} {r2_lat} {r2_long} {ils} {heading}\n")

    total = kept + dropped
    print(f"\nkept {kept} of {total} runway ends, dropped {dropped} ({100.0 * dropped / total:.1f}%) over {MAX_BEARING_MISMATCH} deg")
