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


conf_path = (Path(__file__).resolve() / '../../config.yaml').resolve()
conf = OmegaConf.load(conf_path)

# === DAT FILES ===
AIRPORT_DAT_PATH    = conf.dats.airport_dat_path
NAV_DAT_PATH        = conf.dats.nav_dat_path
RUNWAYS_DAT_PATH      = conf.dats.runways_dat_path

if __name__ == '__main__':
    runways = get_runways(AIRPORT_DAT_PATH, NAV_DAT_PATH)
    with open(RUNWAYS_DAT_PATH, 'w') as fp:
        for runway in runways:
            airport_name, runway_name = runway[0:2]
            elev, runway_width = runway[2]['elev'], runway[2]['runway_width']
            r1_name, r1_lat, r1_long = runway[2]['r1_name'], runway[2]['r1_lat'], runway[2]['r1_long']
            r2_name, r2_lat, r2_long = runway[2]['r2_name'], runway[2]['r2_lat'], runway[2]['r2_long']
            ils, heading = runway[2]['ils'], runway[2]['heading']
            fp.write(f"{airport_name} {runway_name} {elev} {runway_width} {r1_name} {r1_lat} {r1_long} {r2_name} {r2_lat} {r2_long} {ils} {heading}\n")
