import os
import sys
import time
import math
import random
import shutil
from omegaconf import OmegaConf
from XPPython3 import xp
from pathlib import Path
from contextlib import suppress
from timezonefinder import TimezoneFinder
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.dat_utils import parse_runways_dat
from utils.general_utils import (
    degrees_to_rads,
    degrees_to_meters,
    get_start_position,
    euler_to_quaternion,
    mat4_mul_vec4,
    project_local_to_pixel,
    get_runway_points,
    clip_polygon_to_screen,
    clip_near_plane,
    compute_bearing
)



# ============================================================================
# CONFIGURATION
# 
# Global variables used to control flight behavior. Read from conf_path for
# defaults. Avoid overwriting these; they're meant to be overridden ever time
# we switch airports.
# ============================================================================

conf_path = Path(__file__).resolve().parent / 'config.yaml'
conf = OmegaConf.load(conf_path)
dc, ap, desc, env = conf.data_collection, conf.airport, conf.descent, conf.environment

# === AUTOSTART ===
AUTOSTART           = dc.autostart

# === DATASET COLLECTION ===
RUNWAYS_DAT_PATH    = conf.dats.runways_dat_path
SS_PATH             = dc.screenshots_path
OUTPUT_DIR          = dc.output_dir
MAX_RUNWAYS         = dc.max_runways
EPISODES_PER_LAUNCH = dc.episodes_per_launch
FPS                 = dc.fps
SIM_SPEED           = dc.sim_speed

# === AIRPORT ===
AIRPORT             = ap.airport_name
RUNWAY_NAME         = ap.runway_name
RUNWAY_WIDTH        = ap.runway_width
R1_LAT              = ap.r1_lat
R1_LONG             = ap.r1_long
R2_LAT              = ap.r2_lat
R2_LONG             = ap.r2_long
HEADING             = ap.heading
ELEV                = ap.elev
ILS_FREQ            = ap.ils_freq
# this needs to be computed
RUNWAY_POINTS       = get_runway_points(R1_LAT, R1_LONG, R2_LAT, R2_LONG, RUNWAY_WIDTH, ELEV)

# === DESCENT ===
START_DIST          = desc.start_dist
DESCENT_SPEED       = desc.descent_speed
DESCENT_ANGLE       = desc.descent_angle
# this needs to be computed
START_LAT, START_LONG, START_ALT= get_start_position(
    R1_LAT, R1_LONG, HEADING, ELEV, START_DIST, DESCENT_ANGLE
)
# allow START_ALT to be overridden since sometimes the AI fails
if conf.descent.start_altitude:
    START_ALT = conf.descent.start_altitude

# === WIND ===
MAX_WIND_SPEED      = env.wind.max_wind_speed
MAX_GUST_SPEED      = env.wind.max_gust_speed
MAX_TURB            = env.wind.max_turbulence

# === VISIBILITY ===
MIN_VIS             = env.visibility.min_visibility
CLEAR_CHANCE        = env.visibility.clear_chance

# === CLOUDS ===
NUM_CLOUD_LAYERS    = env.clouds.num_layers
CLOUD_BASES         = list(env.clouds.bases)
CLOUD_TOPS          = list(env.clouds.tops)
CLOUD_COVERAGE      = list(env.clouds.coverage)

# === RAIN ===
MAX_RAIN            = env.rain.max_rain
RAIN_CHANCE         = env.rain.rain_chance

# === TIME OF DAY ===
tf = TimezoneFinder()
DAYTIME_CHANCE      = env.time.daytime_chance
# this needs to be computed
TIMEZONE = tf.timezone_at(lat=R1_LAT, lng=R1_LONG)
TIMEZONE_OFFSET= datetime.now(ZoneInfo(TIMEZONE)).utcoffset().total_seconds() / 3600

# ================================================================
# PLUGIN IMPLEMENTATION
# ================================================================
class PythonInterface:

    def __init__(self):
        self.dr = {}
        self.cmd = {}
        self.menu_id = None 
        
        self._runways = []
        self._runway_idx = 0
        self._run_dir = ''
        self._mon_accum = 0
        self._mon_condition_logger = {}
        self._mon_pose_logger = []
        self._mon_position_logger = []
        self._mon_poly_logger = []
        self._is_running = False
        self._world_mat = [0.0] * 16
        self._proj_mat = [0.0] * 16
        self._episodes_this_launch = 0

    def __reset_monitor__(self):
        self._mon_accum = 0
        self._mon_pose_logger = []
        self._mon_position_logger = []
        self._mon_poly_logger = []


    def XPluginStart(self):
        return (
            "XPPython3 Landing Dataset Collector",
            "com.data_collection.landing",
            "Collect runway landing sequences"
        )
    # ============================================================
    # DATAREFS AND COMMANDS
    # 
    # Refer to https://developer.x-plane.com/datarefs/ and
    # https://siminnovations.com/xplane/command/index.php
    # ============================================================
    def XPluginEnable(self):
        refs = {
            # OpenGL coordinates
            "lx":                   "sim/flightmodel/position/local_x",
            "ly":                   "sim/flightmodel/position/local_y",
            "lz":                   "sim/flightmodel/position/local_z",
            
            # quaternion in OpenGL for the physics engine
            "q":                    "sim/flightmodel/position/q",

            # world-frame Euler angles
            "psi":                  "sim/flightmodel/position/psi",
            "theta":                "sim/flightmodel/position/theta",
            "phi":                  "sim/flightmodel/position/phi",
            
            # velocity in local frame
            "vx":                   "sim/flightmodel/position/local_vx",
            "vy":                   "sim/flightmodel/position/local_vy",
            "vz":                   "sim/flightmodel/position/local_vz",
            
            # roll, pitch, and yaw rotation rates in local frame
            "P":                    "sim/flightmodel/position/P",
            "Q":                    "sim/flightmodel/position/Q",
            "R":                    "sim/flightmodel/position/R",
            
            # monitoring air and groundspeed
            "agl":                  "sim/flightmodel/position/y_agl",
            "gnd":                  "sim/flightmodel/failures/onground_any",
            "ias":                  "sim/flightmodel/position/indicated_airspeed",

            # autopilot monitoring; we need to force autopilot the whole time to control gliding
            "ap":                   "sim/cockpit/autopilot/autopilot_mode",
            
            # autopilot and plane control
            "ap_hdg":               "sim/cockpit/autopilot/heading_mag",
            "thro_cmd":             "sim/cockpit2/engine/actuators/throttle_ratio_all",
            "override_joystick":    "sim/operation/override/override_joystick",
            "yoke_roll":            "sim/cockpit2/controls/yoke_roll_ratio",
            "yoke_pitch":           "sim/cockpit2/controls/yoke_pitch_ratio",
            "yoke_heading":         "sim/cockpit2/controls/yoke_heading_ratio",
            
            # radio sources for cockpit setup; not sure why but autopilot needs some of this
            "nav1":                 "sim/cockpit/radios/nav1_freq_hz",
            "nav1_obs":             "sim/cockpit/radios/nav1_obs_degm",
            "hsi_src":              "sim/cockpit2/radios/actuators/HSI_source_select_pilot",

            # weather control
            "weather_source":       "sim/weather/region/weather_source",
            "update_immediately":   "sim/weather/region/update_immediately",
            
            # wind control
            "wind_speed":           "sim/weather/region/wind_speed_msc",
            "wind_direction":       "sim/weather/region/wind_direction_degt",
            "wind_turbulence":      "sim/weather/region/turbulence",

            # visibility control
            "visibility":           "sim/weather/region/visibility_reported_sm",

            # cloud control
            "cloud_base":           "sim/weather/region/cloud_base_msl_m",
            "cloud_tops":           "sim/weather/region/cloud_tops_msl_m",
            "cloud_coverage":       "sim/weather/region/cloud_coverage_percent",

            # rain control
            "rain_pct":             "sim/weather/region/rain_percent",
            
            # time of day; need to shift from UTC
            "zulu_time":            "sim/time/zulu_time_sec",
            "use_sys_time":         "sim/time/use_system_time",

            # world to camera transformations
            "world_matrix":         "sim/graphics/view/world_matrix",
            "proj_matrix":          "sim/graphics/view/projection_matrix_3d",
            "screen_w":             "sim/graphics/view/window_width",
            "screen_h":             "sim/graphics/view/window_height",

            # speed up the sim for data collection purposes
            "sim_speed":            "sim/time/sim_speed",
            "sim_speed_actual" :    "sim/time/sim_speed_actual"
        }

        for key, path in refs.items():
            self.dr[key] = xp.findDataRef(path)
            if self.dr[key] is None:
                xp.log(f"WARN: dataref not found: {path}")

        cmds = {
            # autopilot
            "servos":               "sim/autopilot/servos_on",
            "approach":             "sim/autopilot/approach",

            # weather control
            "regen_weather":        "sim/operation/regen_weather",

            # screen capture
            "toggle_movie":         "sim/operation/video_record_toggle",

            # quit
            "quit":                 "sim/operation/quit",
        }

        for key, path in cmds.items():
            self.cmd[key] = xp.findCommand(path)
            if self.cmd[key] is None:
                xp.log(f" WARN: command not found: {path}")
        
        # keep track of transformation matrices during draw phase
        xp.registerDrawCallback(self._draw_cb, phase=xp.Phase_Window, after=0, refCon=None)

        # make interactable button in X-Plane
        idx = xp.appendMenuItem(xp.findPluginsMenu(), "Begin Data Collection", 0)
        self.menu_id = xp.createMenu("Begin Data Collection", xp.findPluginsMenu(), idx, self._menu_cb)
        xp.appendMenuItem(self.menu_id, "Begin Data Collection", "go")
        xp.log(" Plugin enabled.")
        xp.registerFlightLoopCallback(self._autostart_cb, interval=10.0)
        return 1



    def XPluginDisable(self):
        if self.menu_id:
            xp.unregisterDrawCallback(self._draw_cb, phase=xp.Phase_Window, after=0, refCon=None)
            xp.destroyMenu(self.menu_id)
            self.menu_id = None



    def XPluginStop(self):
        pass



    # ============================================================
    # MESSAGES
    # ============================================================
    def XPluginReceiveMessage(self, fromWho, message, param):
        pass


    # ============================================================
    # ENVIRONMENT CONTROL
    # ============================================================
    def _randomize_wind(self):    
        # Random surface wind
        base_speed = random.uniform(0.0, MAX_WIND_SPEED)
        base_dir = random.uniform(0.0, 360.0)
        gust_extra = random.uniform(0.0, MAX_GUST_SPEED)
        turb = random.uniform(0.0, MAX_TURB)
    
        # Fill all layers with similar values (slight increase with altitude)
        speeds = [base_speed + gust_extra + i * 0.3 for i in range(13)]
        dirs = [base_dir for _ in range(13)]
        turbs = [turb for _ in range(13)]
    
        xp.setDatavf(self.dr["wind_speed"], speeds, 0, 13)
        xp.setDatavf(self.dr["wind_direction"], dirs, 0, 13)
        xp.setDatavf(self.dr["wind_turbulence"], turbs, 0, 13)
        
        self._mon_condition_logger['wind'] = [base_speed, base_dir, turb]



    def _randomize_visibility(self):
        if random.random() < CLEAR_CHANCE:
            vis = 15.0
        else:
            vis = random.uniform(MIN_VIS, 15.0)
        xp.setDataf(self.dr["visibility"], vis)
        self._mon_condition_logger['vis'] = vis



    def _set_clouds(self):
        xp.setDatavf(self.dr["cloud_base"], CLOUD_BASES, 0, NUM_CLOUD_LAYERS)
        xp.setDatavf(self.dr["cloud_tops"], CLOUD_TOPS, 0, NUM_CLOUD_LAYERS)
        xp.setDatavf(self.dr["cloud_coverage"], CLOUD_COVERAGE, 0, NUM_CLOUD_LAYERS)



    def _randomize_precipitation(self):
        
        if random.random() < RAIN_CHANCE:
            rain = random.uniform(MAX_RAIN/2, MAX_RAIN)
        else:
            rain = 0.0
        xp.setDataf(self.dr["rain_pct"], rain)
        self._mon_condition_logger['rain'] = rain



    def _randomize_time(self):
        # Disable system time so our writes stick
        xp.setDatai(self.dr["use_sys_time"], 0)
    
        if random.random() < DAYTIME_CHANCE:
            # Daytime: 8am - 5pm local
            local_sec = random.uniform(28800, 61200)
        else:
            # Nighttime: 8pm - 5am local
            if random.random() < 0.5:
                local_sec = random.uniform(72000, 86400)
            else:
                local_sec = random.uniform(0, 18000)
    
        zulu_sec = (local_sec - TIMEZONE_OFFSET * 3600) % 86400
        xp.setDataf(self.dr["zulu_time"], zulu_sec)
    
        hours = int(local_sec // 3600)
        mins = int((local_sec % 3600) // 60)
        self._mon_condition_logger['daytime'] = 'true' if 28800 <= local_sec <= 61200 else 'false'
   


    def _count_completed_episodes(self, output_dir):
        """Count dirs with a 'poly.txt' sentinel — written last in _flare_loop."""
        if not os.path.isdir(output_dir):
            return 0
        count = 0
        for name in os.listdir(output_dir):
            d = os.path.join(output_dir, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'poly.txt')):
                count += 1
        return count


    def _draw_cb(self, phase, after, refcon):
        # Cache the matrices every frame
        self._world_mat = [0.0] * 16
        xp.getDatavf(self.dr["world_matrix"], self._world_mat, 0, 16)
        self._proj_mat = [0.0] * 16
        xp.getDatavf(self.dr["proj_matrix"], self._proj_mat, 0, 16)
        self._sw = xp.getDatai(self.dr["screen_w"])
        self._sh = xp.getDatai(self.dr["screen_h"])
        return 1

    # ============================================================
    # AUTOSTART AND QUIT (FOR XP REFRESHES)
    # ============================================================
    def _autostart_cb(self, since_last, elapsed, counter, refcon):
        if AUTOSTART:
            self._menu_cb(None, "go")
        return 0
    def _quit_cb(self, since_last, elapsed, counter, refcon):
        xp.commandOnce(self.cmd["quit"])
        return 0



    # ============================================================
    # START BEHAVIOR
    # ============================================================
    def _menu_cb(self, menuRef, itemRef):
        if itemRef == "go" and self._is_running == False:
            try:
                self._is_running = True
                
                xp.log(" === Setting up output directory ===")
                
                # make output directory, replace screenshot with symlink to output
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                p = Path(SS_PATH)
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.exists():
                    shutil.rmtree(p)
                os.symlink(OUTPUT_DIR, SS_PATH, target_is_directory=True)

                xp.log(" === Retrieve runway data ===")
                runways = parse_runways_dat(RUNWAYS_DAT_PATH)
                max_runways = min(len(runways), MAX_RUNWAYS)
                xp.log(f"Found {len(runways)} runways, using {max_runways} runways")
                self._runways = runways[:max_runways]

                # check for start point
                completed = self._count_completed_episodes(OUTPUT_DIR)
                xp.log(f"Found {completed} completed episodes on disk")
                self._runway_idx = completed

                if completed >= max_runways:
                    xp.log(" === Dataset complete ===")
                    Path(OUTPUT_DIR, "DONE").touch()
                    xp.commandOnce(self.cmd["quit"])
                    return
                
                # set up clouds + rain and increase sim speed
                self._set_clouds()
                self._randomize_precipitation()
                xp.setDatai(self.dr["weather_source"], 1)
                xp.setDatai(self.dr["update_immediately"], 1)
                xp.commandOnce(self.cmd["regen_weather"])
                xp.setDatai(self.dr["sim_speed"], SIM_SPEED)
                
                # loop for all runways
                xp.log(" === Begin flight loop ===")
                self._start_next_flight()
            except Exception as e:
                import traceback
                xp.log(f"ERROR: {e}\n{traceback.format_exc()}")

    # ============================================================
    # DESCENT FLARE LOOP
    # ============================================================
    def _start_next_flight(self):
        # check if we are done; else, fetch the runway
        if self._runway_idx >= len(self._runways):
            xp.log(" === All runways complete ===")
            self._is_running = False
            return
        r = self._runways[self._runway_idx]
        counter = self._runway_idx + 1

        # per-launch restart check
        if self._episodes_this_launch >= EPISODES_PER_LAUNCH:
            xp.log(f" === Hit {EPISODES_PER_LAUNCH} episodes this launch, restarting ===")
            self._is_running = False
            # let any pending writes settle
            xp.registerFlightLoopCallback(self._quit_cb, interval=3.0)
            return
        
        # reference global variables
        global AIRPORT, RUNWAY_NAME, RUNWAY_WIDTH
        global R1_LAT, R1_LONG, R2_LAT, R2_LONG
        global HEADING, ELEV, ILS_FREQ
        global RUNWAY_POINTS, START_LAT, START_LONG, START_ALT
        global TIMEZONE, TIMEZONE_OFFSET

        # update global variables based on new parameters
        AIRPORT = r['airport_name']
        RUNWAY_NAME = r['runway_name']
        RUNWAY_WIDTH = r['runway_width']
        R1_LAT = r['r1_lat']
        R1_LONG = r['r1_long']
        R2_LAT = r['r2_lat']
        R2_LONG = r['r2_long']
        HEADING = r['heading']
        ELEV = r['elev']
        ILS_FREQ = r['ils']
        
        # these needs to be computed
        RUNWAY_POINTS = get_runway_points(R1_LAT, R1_LONG, R2_LAT, R2_LONG, RUNWAY_WIDTH, ELEV)
        START_LAT, START_LONG, START_ALT= get_start_position(
            R1_LAT, R1_LONG, HEADING, ELEV, START_DIST, DESCENT_ANGLE
        )
        if conf.descent.start_altitude:
            START_ALT = conf.descent.start_altitude
        TIMEZONE = tf.timezone_at(lat=R1_LAT, lng=R1_LONG) or "UTC"
        TIMEZONE_OFFSET = datetime.now(ZoneInfo(TIMEZONE)).utcoffset().total_seconds() / 3600
            
        # update location symlinks to point to idx
        self._run_dir = os.path.join(OUTPUT_DIR, f"{counter:06}")
        os.makedirs(self._run_dir, exist_ok=True)
        p = Path(SS_PATH)
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.exists():
            shutil.rmtree(p)
        os.symlink(self._run_dir, SS_PATH, target_is_directory=True)

        # reset monitor variables
        self.__reset_monitor__()

        # Randomize environment conditions (except cloud cover and precipitation)
        self._randomize_wind()
        self._randomize_visibility()
        self._randomize_time()
        xp.setDatai(self.dr["weather_source"], 1)
        xp.setDatai(self.dr["update_immediately"], 1)
        xp.commandOnce(self.cmd["regen_weather"])

        # disable previous controls from previous
        with suppress(Exception): xp.unregisterFlightLoopCallback(self._flare_loop)
        with suppress(Exception): xp.unregisterFlightLoopCallback(self._monitor_loop)
        xp.setDatai(self.dr["override_joystick"], 0)
        
        # delayed placement at the new airport
        xp.placeUserAtAirport(AIRPORT)
        xp.registerFlightLoopCallback(self._place_on_approach, interval=1.0)

    # ============================================================
    # CALLBACK FOR NEXT FLIGHT
    #
    # Calling _start_next_flight directly in _flare_loop seems to
    # occassionally crash XPlane-12 due to race conditions
    # ============================================================
    def _advance_cb(self, since_last, elapsed, counter, refcon):
        self._start_next_flight()
        return 0   # one-shot


    # ============================================================
    # DESCENT FLARE LOOP
    # ============================================================
    def _flare_loop(self, since_last, elapsed, counter, refcon):

        # monitor the flight; once we get too low, we need to flare for the landing sequence
        if not hasattr(self, '_flare_startup'):
            self._flare_startup = 0.0
        self._flare_startup += since_last
        if self._flare_startup < 2.0:
            return -1
        
        agl = xp.getDataf(self.dr["agl"])
        ias = xp.getDataf(self.dr["ias"])
        
        # --- THROTTLE: maintain target speed throughout ---
        if agl > 40:
            # Simple speed control: if too fast reduce throttle, if too slow increase
            speed_error = ias - DESCENT_SPEED
            throttle = 0.5 - (speed_error / 40.0)
            throttle = max(0.0, min(1.0, throttle))
            xp.setDataf(self.dr["thro_cmd"], throttle)
        
        # --- FLARE: below 50ft, take over from AP ---
        if agl < 40:
            xp.setDatai(self.dr["override_joystick"], 1)
            
            # Hold wings level
            phi = xp.getDataf(self.dr["phi"])
            xp.setDataf(self.dr["yoke_roll"], max(-1.0, min(1.0, -phi / 25.0)))
            
            # pitch: ramps from 0.3 at 50ft
            target_pitch = 0.3 + (1.0 - agl/40.0) * 0.2
            # throttle: ramps from 0.2 at 50ft
            target_throttle = 0.2 - (1.0 - agl/40.0) * 0.15
            xp.setDataf(self.dr["yoke_pitch"], target_pitch)
            xp.setDataf(self.dr["thro_cmd"], target_throttle)
        
        # Touchdown
        if agl < 5 or xp.getDatai(self.dr["gnd"]):
            xp.setDatai(self.dr["override_joystick"], 0)
            xp.setDataf(self.dr["thro_cmd"], 0.0)
            xp.commandOnce(self.cmd["toggle_movie"])
            with suppress(Exception): xp.unregisterFlightLoopCallback(self._monitor_loop)
            
            # write monitored conditions to file
            gt_meta_path = os.path.join(self._run_dir, 'meta.txt')
            gt_pose_path = os.path.join(self._run_dir, 'pose.txt')
            gt_position_path = os.path.join(self._run_dir, 'position.txt')
            gt_poly_path = os.path.join(self._run_dir, 'poly.txt')

            with open(gt_meta_path, 'w') as fp:
                fp.write(f'{AIRPORT} {RUNWAY_NAME}\n')
                env_cond = ' '.join([
                    str(self._mon_condition_logger['wind'][0]),
                    str(self._mon_condition_logger['wind'][1]),
                    str(self._mon_condition_logger['wind'][2]),
                    str(self._mon_condition_logger['vis']),
                    str(self._mon_condition_logger['rain']),
                    self._mon_condition_logger['daytime'],
                ])
                fp.write(env_cond + '\n')

            with open(gt_pose_path, 'w') as fp:
                for q in self._mon_pose_logger:
                    fp.write(f'{q[0]} {q[1]} {q[2]} {q[3]}\n')
            
            with open(gt_position_path, 'w') as fp:
                for x, y, z in self._mon_position_logger:
                    fp.write(f'{x} {y} {z}\n')

            with open(gt_poly_path, 'w') as fp:
                for poly in self._mon_poly_logger:
                    poly = [f"{p[0]} {p[1]}" for p in poly]
                    if len(poly) == 0:
                        fp.write(f'\n')
                    else:
                        fp.write(' '.join(poly) + '\n')

            # move onto next flight; slightly delay to avoid race conditions
            self._runway_idx += 1
            self._episodes_this_launch += 1
            xp.registerFlightLoopCallback(self._advance_cb, interval=0.5)
            return 0
        
        return -1



    def _get_runway_polygon(self):
        """
        Project runway corners to screen pixels. Returns list of (x_norm, y_norm) 
        normalized 0-1, or None if runway not visible.
        """
        # Read matrices
        sw = xp.getDatai(self.dr["screen_w"])
        sh = xp.getDatai(self.dr["screen_h"])
    
        if sw <= 0 or sh <= 0:
            return None
    
        # Project each point
        screen_pts = []
        behind_flags = []
        clips = []
        for lat, lon, elev in RUNWAY_POINTS:
            lx, ly, lz = xp.worldToLocal(lat, lon, elev)
            px, py, behind, clip = project_local_to_pixel(lx, ly, lz, self._world_mat, self._proj_mat, sw, sh)
            screen_pts.append((px, py))
            behind_flags.append(behind)
            clips.append(clip)
    
        # If all behind camera, not visible
        if all(behind_flags):
            return None
    
        # Clip near plane
        if any(behind_flags):
            poly = clip_near_plane(clips, screen_pts, behind_flags, sw, sh)
        else:
            poly = list(screen_pts)
    
        # Clip to screen bounds
        poly = clip_polygon_to_screen(poly, sw, sh)
    
        if len(poly) < 3:
            return None
    
        # Normalize to 0-1 (flip Y: screen bottom-left to top-left origin for YOLO)
        normalized = []
        for px, py in poly:
            nx = max(0.0, min(1.0, px / sw))
            ny = max(0.0, min(1.0, 1.0 - py / sh))
            normalized.append((nx, ny))
    
        return normalized



    # ============================================================
    # MONITOR AND COLLECT GT LABELS
    # ============================================================
    def _monitor_loop(self, since_last, elapsed, counter, refcon):
        self._mon_accum += since_last
        agl = xp.getDataf(self.dr["agl"])

        # capture quaternions for rotation
        q = [0.0] * 4
        xp.getDatavf(self.dr['q'], q, 0, 4)

        # capture location
        rx, ry, rz = xp.worldToLocal(R1_LAT, R1_LONG, ELEV)
        x = xp.getDatad(self.dr['lx']) - rx
        y = xp.getDataf(self.dr['agl'])
        z = xp.getDatad(self.dr['lz']) - rz
        
        # converted captured locations from absolute offset to relative to the landing strip
        h = math.radians(HEADING)
        cos_h, sin_h = math.cos(h), math.sin(h)
        _x = x * sin_h - z * cos_h   # positive = past threshold (down the runway)
        _z = x * cos_h + z * sin_h   # positive = right of centerline (facing forward)
        x, z = _x, _z

        # get polygon of runway
        poly = self._get_runway_polygon()

        # write to loggers for future writing
        self._mon_pose_logger.append(q)
        self._mon_position_logger.append([x, y, z])
        if poly:
            self._mon_poly_logger.append(poly)
        else:
            self._mon_poly_logger.append([])

        # else, delay next callback based on intended FPS
        return (1 / FPS)



    def _place_on_approach(self, since_last, elapsed, counter, refcon):
        # Convert both points to X-Plane local coords
        tx, ty, tz = xp.worldToLocal(R1_LAT, R1_LONG, ELEV)
        sx, sy, sz = xp.worldToLocal(START_LAT, START_LONG, START_ALT)
        if conf.descent.start_altitude:
            sy = ty + START_ALT

        # Compute true heading from start to threshold using local coords
        # X-Plane: +X=east, +Y=up, +Z=south
        dx = tx - sx
        dy = ty - sy
        dz = tz - sz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        true_heading = math.degrees(math.atan2(dx, -dz)) % 360.0
        horiz_dist = math.sqrt(dx*dx + dz*dz)
        pitch = -math.degrees(math.atan2(-dy, horiz_dist))

        # === POSITION ===
        xp.setDatad(self.dr["lx"], sx)
        xp.setDatad(self.dr["ly"], sy)
        xp.setDatad(self.dr["lz"], sz)

        # === ORIENTATION ===
        q = euler_to_quaternion(true_heading, pitch, 0.0)
        xp.setDatavf(self.dr["q"], q)
        xp.setDataf(self.dr["psi"], true_heading)
        xp.setDataf(self.dr["theta"], pitch)
        xp.setDataf(self.dr["phi"], 0.0)

        # === VELOCITY toward threshold ===
        speed_ms = DESCENT_SPEED * 0.5144
        xp.setDataf(self.dr["vx"], speed_ms * dx / dist)
        xp.setDataf(self.dr["vy"], speed_ms * dy / dist)
        xp.setDataf(self.dr["vz"], speed_ms * dz / dist)
        xp.setDataf(self.dr["P"], 0.0)
        xp.setDataf(self.dr["Q"], 0.0)
        xp.setDataf(self.dr["R"], 0.0)

        # === THROTTLE ===
        xp.setDataf(self.dr["thro_cmd"], 0.7)

        # === RADIOS ===
        xp.setDatai(self.dr["hsi_src"], 0)
        xp.setDatai(self.dr["nav1"], ILS_FREQ)
        xp.setDataf(self.dr["nav1_obs"], HEADING)

        # === AUTOPILOT ===
        xp.setDataf(self.dr["ap_hdg"], HEADING)
        xp.setDatai(self.dr["ap"], 2)
        xp.commandOnce(self.cmd["servos"])
        xp.commandOnce(self.cmd["approach"])

        # === FLARE LOOP === #
        self._flare_startup = 0.0
        xp.registerFlightLoopCallback(self._flare_loop, interval=-1)

        # === RECORDING === #
        xp.commandOnce(self.cmd["toggle_movie"])

        # === MONITOR LOOP ===
        xp.registerFlightLoopCallback(self._monitor_loop, interval=-1)

        return 0
