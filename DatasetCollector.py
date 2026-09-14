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
    get_start_position,
    euler_to_quaternion,
    mat4_mul_vec4,
    project_local_to_pixel,
    get_runway_corners,
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
ELEV                = ap.elev
ILS_FREQ            = ap.ils_freq
# these need to be computed.
# RUNWAY_AXIS is the bearing of the pavement, derived from the two runway-end
# coordinates -- NOT the ILS localiser bearing in the dat file. Everything we log is
# measured against the runway the labels describe, so the two must not be mixed.
# dat_process.py guarantees they agree to within MAX_BEARING_MISMATCH degrees.
RUNWAY_AXIS         = compute_bearing(R1_LAT, R1_LONG, R2_LAT, R2_LONG)
# the two runway-end elevations get replaced by terrain probes once scenery is loaded;
# the published airport elevation is only the fallback
RUNWAY_CORNERS      = get_runway_corners(R1_LAT, R1_LONG, R2_LAT, R2_LONG, RUNWAY_WIDTH, ELEV, ELEV)

# === DESCENT ===
START_DIST          = desc.start_dist
STOP_DIST           = desc.stop_dist
DESCENT_SPEED       = desc.descent_speed
DESCENT_ANGLE       = desc.descent_angle
FLARE_ALT           = desc.flare_alt
FLARE_START_PITCH   = desc.flare_start_pitch
FLARE_END_PITCH     = desc.flare_end_pitch
TOUCHDOWN_ALT       = desc.touchdown_alt
# this needs to be computed
START_LAT, START_LONG, START_ALT= get_start_position(
    R1_LAT, R1_LONG, RUNWAY_AXIS, ELEV, START_DIST, DESCENT_ANGLE
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
        self._probe = None
        self._thr_y = 0.0          # OpenGL y of the runway surface at the threshold
        self._capture_done = False

    def __reset_monitor__(self):
        self._mon_accum = 0
        self._mon_pose_logger = []
        self._mon_position_logger = []
        self._mon_poly_logger = []
        self._capture_done = False


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
            
            # Euler-angle rotation rates; for some reason if I don't register these, the plane just crashes???
            # DO NOT REMOVE???
            "P":                    "sim/flightmodel/position/P",
            "Q":                    "sim/flightmodel/position/Q",
            "R":                    "sim/flightmodel/position/R",
            
            # velocity in local frame
            "vx":                   "sim/flightmodel/position/local_vx",
            "vy":                   "sim/flightmodel/position/local_vy",
            "vz":                   "sim/flightmodel/position/local_vz",
             
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
            # weather_source is READ-ONLY, so writing it does nothing. change_mode is the
            # writable one; 3 = Static, which stops the sim drifting our values back
            "change_mode":          "sim/weather/region/change_mode",
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

        # terrain probe, used to put the runway polygon on the actual ground rather than
        # on the airport's single published elevation
        self._probe = xp.createProbe()

        # make interactable button in X-Plane
        idx = xp.appendMenuItem(xp.findPluginsMenu(), "Begin Data Collection", 0)
        self.menu_id = xp.createMenu("Begin Data Collection", xp.findPluginsMenu(), idx, self._menu_cb)
        xp.appendMenuItem(self.menu_id, "Begin Data Collection", "go")
        xp.log(" Plugin enabled.")
        xp.registerFlightLoopCallback(self._autostart_cb, interval=10.0)
        return 1



    def XPluginDisable(self):
        if self._probe is not None:
            xp.destroyProbe(self._probe)
            self._probe = None
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
        
        # log the surface layer we actually wrote, gust included -- not base_speed,
        # which is not what any layer ends up holding
        self._mon_condition_logger['wind'] = [speeds[0], base_dir, turb]



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
        """Count dirs with a 'poly.txt' sentinel — written last in _finish_capture."""
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
    # END OF DATASET
    #
    # Raise the sentinel supervisor.bash waits on, then shut the sim
    # down. Without this an unattended run reaches its last episode
    # and then sits idle forever, holding the supervisor loop open.
    # ============================================================
    def _finish_dataset(self):
        xp.log(" === Dataset complete ===")
        self._is_running = False

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            Path(OUTPUT_DIR, "DONE").touch()
        except Exception as e:
            # still quit; a stuck sim is worse than a missing flag
            xp.log(f" WARN: could not write the DONE flag: {e}")

        # delayed so the last episode's writes land before the process goes away
        with suppress(Exception): xp.unregisterFlightLoopCallback(self._quit_cb)
        xp.registerFlightLoopCallback(self._quit_cb, interval=3.0)



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
                    self._finish_dataset()
                    return
                
                # set up clouds and increase sim speed. precipitation is rolled per
                # episode in _start_next_flight, not once per launch
                self._set_clouds()
                xp.setDatai(self.dr["change_mode"], 3)
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
            self._finish_dataset()
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
        global ELEV, ILS_FREQ
        global RUNWAY_AXIS, RUNWAY_CORNERS, START_LAT, START_LONG, START_ALT
        global TIMEZONE, TIMEZONE_OFFSET

        # update global variables based on new parameters
        AIRPORT = r['airport_name']
        RUNWAY_NAME = r['runway_name']
        RUNWAY_WIDTH = r['runway_width']
        R1_LAT = r['r1_lat']
        R1_LONG = r['r1_long']
        R2_LAT = r['r2_lat']
        R2_LONG = r['r2_long']
        ELEV = r['elev']
        ILS_FREQ = r['ils']

        # these needs to be computed.
        # r['heading'] is the ILS localiser bearing and is deliberately not used here;
        # the pavement bearing is what the ground truth is measured against
        RUNWAY_AXIS = compute_bearing(R1_LAT, R1_LONG, R2_LAT, R2_LONG)
        # provisional corners at the published elevation; _place_on_approach re-derives
        # them from terrain probes once the scenery around the airport has loaded
        RUNWAY_CORNERS = get_runway_corners(R1_LAT, R1_LONG, R2_LAT, R2_LONG, RUNWAY_WIDTH, ELEV, ELEV)
        START_LAT, START_LONG, START_ALT= get_start_position(
            R1_LAT, R1_LONG, RUNWAY_AXIS, ELEV, START_DIST, DESCENT_ANGLE
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

        # Randomize environment conditions (except cloud cover)
        self._randomize_wind()
        self._randomize_visibility()
        self._randomize_precipitation()
        self._randomize_time()
        xp.setDatai(self.dr["change_mode"], 3)
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
        
        ias = xp.getDataf(self.dr["ias"])

        # Height above the RUNWAY, not above whatever terrain is underneath. y_agl drops
        # as terrain rises on the approach, which used to trigger the flare a kilometre
        # out -- throttle to idle, nose up, and the aircraft settled into the trees short
        # of the runway. Real ground contact is still caught by the onground_any test in
        # the touchdown branch below, so an approach that does hit terrain still ends.
        agl = xp.getDatad(self.dr["ly"]) - self._thr_y

        # --- THROTTLE: maintain target speed throughout ---
        if agl > FLARE_ALT:
            # Simple speed control: if too fast reduce throttle, if too slow increase
            speed_error = ias - DESCENT_SPEED
            throttle = 0.5 - (speed_error / 40.0)
            throttle = max(0.0, min(1.0, throttle))
            xp.setDataf(self.dr["thro_cmd"], throttle)
        
        # --- FLARE: below FLARE_ALT take over from AP ---
        if agl < FLARE_ALT and agl >= TOUCHDOWN_ALT:
            xp.setDatai(self.dr["override_joystick"], 1)
            
            # Hold wings level
            phi = xp.getDataf(self.dr["phi"])
            xp.setDataf(self.dr["yoke_roll"], max(-1.0, min(1.0, -phi / 25.0)))
            
            # Flare progress: 0.0 at FLARE_ALT, 1.0 at ground
            progress = 1.0 - (agl / FLARE_ALT)
            progress = max(0.0, min(1.0, progress))
            
            # Pitch ramps up gently
            target_pitch = FLARE_START_PITCH + progress * (FLARE_END_PITCH - FLARE_START_PITCH)
            xp.setDataf(self.dr["yoke_pitch"], target_pitch)
            
            # Throttle smoothly to idle (no hard step)
            target_throttle = 0.15 * (1.0 - progress)
            xp.setDataf(self.dr["thro_cmd"], target_throttle)


        # Touchdown
        if agl < TOUCHDOWN_ALT or xp.getDatai(self.dr["gnd"]):
            xp.setDatai(self.dr["override_joystick"], 0)
            xp.setDataf(self.dr["thro_cmd"], 0.0)

            # capture normally ended back at STOP_DIST. this covers the aircraft that
            # never got that close: landed short, or the approach fell apart
            if not self._capture_done:
                xp.log(" WARN: touched down before reaching the capture cutoff")
                self._finish_capture()

            # move onto next flight; slightly delay to avoid race conditions
            self._runway_idx += 1
            self._episodes_this_launch += 1
            xp.registerFlightLoopCallback(self._advance_cb, interval=0.5)
            return 0

        return -1



    # ============================================================
    # END OF CAPTURE
    #
    # The recorder and the ground-truth stream MUST stop on the same
    # event: convert_to_yolo.py pairs them by taking the last N of
    # each, so any drift between the two stop points silently shifts
    # every frame/label pairing in the episode.
    # ============================================================
    def _finish_capture(self):
        if self._capture_done:
            return
        self._capture_done = True

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

        # columns are: yaw pitch roll
        with open(gt_pose_path, 'w') as fp:
            for yaw, pitch, roll in self._mon_pose_logger:
                # positive yaw   -> nose right of the runway heading
                # positive pitch -> nose up
                # positive roll  -> right wing down
                fp.write(f'{yaw} {pitch} {roll}\n')

        # columns are: along height right
        with open(gt_position_path, 'w') as fp:
            for along, height, right in self._mon_position_logger:
                # along  -> meters past the threshold, negative on approach
                # height -> meters above the runway surface at the threshold
                # right  -> meters right of the centerline, facing down the runway
                fp.write(f'{along} {height} {right}\n')

        # written last: _count_completed_episodes uses this file as the resume sentinel
        with open(gt_poly_path, 'w') as fp:
            for poly in self._mon_poly_logger:
                poly = [f"{p[0]} {p[1]}" for p in poly]
                if len(poly) == 0:
                    fp.write(f'\n')
                else:
                    fp.write(' '.join(poly) + '\n')



    def _get_runway_polygon(self):
        """
        Project the four runway corners to normalized image coordinates, in boundary
        order (near-left, far-left, far-right, near-right).

        No polygon clipping: capture stops at STOP_DIST precisely so the whole rectangle
        stays inside the frame, which is also what makes a 4-point label sufficient.
        Returns None only if a corner has gone behind the camera.
        """
        sw = xp.getDatai(self.dr["screen_w"])
        sh = xp.getDatai(self.dr["screen_h"])

        if sw <= 0 or sh <= 0:
            return None

        normalized = []
        for lat, lon, elev in RUNWAY_CORNERS:
            lx, ly, lz = xp.worldToLocal(lat, lon, elev)
            px, py, behind = project_local_to_pixel(lx, ly, lz, self._world_mat, self._proj_mat, sw, sh)
            if behind:
                return None
            # normalize to 0-1, flipping Y from OpenGL bottom-left to YOLO top-left origin
            nx = max(0.0, min(1.0, px / sw))
            ny = max(0.0, min(1.0, 1.0 - py / sh))
            normalized.append((nx, ny))

        return normalized



    # ============================================================
    # MONITOR AND COLLECT GT LABELS
    # ============================================================
    def _monitor_loop(self, since_last, elapsed, counter, refcon):
        self._mon_accum += since_last

        # capture yaw, pitch, roll for rotation
        yaw   = (xp.getDataf(self.dr["psi"]) - RUNWAY_AXIS + 540.0) % 360.0 - 180.0
        pitch = xp.getDataf(self.dr["theta"])
        roll  = xp.getDataf(self.dr["phi"])

        # capture location as an offset from the threshold, in OpenGL local coords
        rx, ry, rz = xp.worldToLocal(R1_LAT, R1_LONG, ELEV)
        dx = xp.getDatad(self.dr['lx']) - rx
        dz = xp.getDatad(self.dr['lz']) - rz
        # height above the runway surface. NOT y_agl, which measures against whatever
        # terrain happens to be under the aircraft and wanders off the glidepath
        height = xp.getDatad(self.dr['ly']) - self._thr_y

        # convert the raw east/south offset into the runway frame
        h = math.radians(RUNWAY_AXIS)
        cos_h, sin_h = math.cos(h), math.sin(h)
        along = dx * sin_h - dz * cos_h   # positive = past threshold (down the runway)
        right = dx * cos_h + dz * sin_h   # positive = right of centerline (facing forward)

        # stop capturing before the near corners start dropping out of frame. the flare
        # loop keeps flying and lands the aircraft, just without recording it
        if along > -STOP_DIST:
            self._finish_capture()
            return 0

        # get polygon of runway
        poly = self._get_runway_polygon()

        # write to loggers for future writing
        self._mon_pose_logger.append([yaw, pitch, roll])
        self._mon_position_logger.append([along, height, right])
        self._mon_poly_logger.append(poly if poly else [])

        # else, delay next callback based on intended FPS
        return (1 / FPS)



    def _probe_surface(self, lat, lon, fallback_elev):
        """
        Find the actual ground at a lat/lon. apt.dat publishes a single elevation for the
        whole airport, which can sit tens of meters off the real runway surface; probing
        fixes both the polygon's altitude and the height we log.

        [out]
            local_y (float): OpenGL y of the ground
            elev (float): the same point as an MSL elevation in meters
        """
        px, py, pz = xp.worldToLocal(lat, lon, fallback_elev)
        if self._probe is not None:
            info = xp.probeTerrainXYZ(self._probe, px, py, pz)
            if info is not None and info.result == xp.ProbeHitTerrain:
                # local y is linear in MSL elevation over a patch this small, so the
                # offset between probed and requested y carries straight over to MSL
                return info.locationY, fallback_elev + (info.locationY - py)
        xp.log(f" WARN: terrain probe missed at {lat},{lon}; using published elevation")
        return py, fallback_elev


    def _place_on_approach(self, since_last, elapsed, counter, refcon):
        global RUNWAY_CORNERS

        # scenery is loaded by now, so pin the runway to the ground it is actually on
        self._thr_y, thr_elev = self._probe_surface(R1_LAT, R1_LONG, ELEV)
        _, far_elev = self._probe_surface(R2_LAT, R2_LONG, ELEV)
        RUNWAY_CORNERS = get_runway_corners(
            R1_LAT, R1_LONG, R2_LAT, R2_LONG, RUNWAY_WIDTH, thr_elev, far_elev
        )

        # Convert both points to X-Plane local coords. The threshold's height comes from
        # the probe, not apt.dat: the published airport elevation can be tens of metres
        # off the runway surface, and measuring start_altitude from it placed BGTL 08T
        # 77 m too high -- 104 m above its glideslope, which it never recovered from.
        tx, _, tz = xp.worldToLocal(R1_LAT, R1_LONG, ELEV)
        ty = self._thr_y
        sx, _, sz = xp.worldToLocal(START_LAT, START_LONG, START_ALT)

        # START_ALT is a height above the runway when overridden, and MSL otherwise;
        # either way we want the height above the probed surface
        start_height = START_ALT if conf.descent.start_altitude else (START_ALT - ELEV)
        sy = ty + start_height

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
        # heading_mag and nav1_obs_degm are both degrees MAGNETIC; RUNWAY_AXIS is true.
        # the conversion uses the user's current location, which is this airport by now
        runway_axis_mag = xp.degTrueToDegMagnetic(RUNWAY_AXIS)
        xp.setDatai(self.dr["hsi_src"], 0)
        xp.setDatai(self.dr["nav1"], ILS_FREQ)
        xp.setDataf(self.dr["nav1_obs"], runway_axis_mag)

        # === AUTOPILOT ===
        xp.setDataf(self.dr["ap_hdg"], runway_axis_mag)
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
