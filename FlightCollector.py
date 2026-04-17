import os
import sys
import time
import math
import random
from omegaconf import OmegaConf
from XPPython3 import xp
from pathlib import Path
from contextlib import suppress

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.dat_utils import get_runways
from utils.general_utils import (
    degrees_to_rads,
    degrees_to_meters,
    get_start_position,
    euler_to_quaternion,
    mat4_mul_vec4,
    project_local_to_pixel,
    get_runway_points,
    clip_polygon_to_screen,
    clip_near_plane
)



# ============================================================================
# CONFIGURATION
# 
# Global variables used to control flight behavior. Read from conf_path for
# defaults. Avoid overwriting these; they're meant to be overridden ever time
# we switch airports.
# ============================================================================

conf_path = (Path(__file__).resolve() / '../config.yaml').resolve()
conf = OmegaConf.load(conf_path)
ap, desc, env = conf.airport, conf.descent, conf.environment

# === DAT FILES ===
AIRPORT_DAT_PATH    = conf.airport_dat_path
NAV_DAT_PATH        = conf.nav_dat_path

# === AIRPORT ===
AIRPORT             = ap.airport_name
RUNWAY_LENGTH       = ap.runway_length
RUNWAY_WIDTH        = ap.runway_width
R1_LAT              = ap.r1_lat
R1_LONG             = ap.r1_long
R2_LAT              = ap.r2_lat
R2_LONG             = ap.r2_long
HEADING             = ap.heading
ELEV                = ap.elev
ILS_FREQ            = ap.ils_freq
# this needs to be computed
RUNWAY_POINTS       = get_runway_points(R1_LAT, R1_LONG, R2_LAT, R2_LONG, RUNWAY_WIDTH)

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
DAYTIME_CHANCE      = env.time.daytime_chance
TIMEZONE_OFFSET     = env.time.timezone_offset


# ================================================================
# PLUGIN IMPLEMENTATION
# ================================================================
class PythonInterface:

    def __init__(self):
        self.dr = {}
        self.cmd = {}
        self.menu_id = None 
        self._placement_pending = False

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
            "ap_state":             "sim/cockpit/autopilot/autopilot_state",
            
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
                xp.log(f"[DC] WARN: dataref not found: {path}")

        cmds = {
            # autopilot
            "servos":               "sim/autopilot/servos_on",
            "approach":             "sim/autopilot/approach",

            # weather control
            "regen_weather":        "sim/operation/regen_weather",

            # screen capture
            "toggle_movie":         "sim/operation/video_record_toggle",
        }

        for key, path in cmds.items():
            self.cmd[key] = xp.findCommand(path)
            if self.cmd[key] is None:
                xp.log(f"[DC] WARN: command not found: {path}")

        # make interactable button in X-Plane
        idx = xp.appendMenuItem(xp.findPluginsMenu(), "Begin Data Collection", 0)
        self.menu_id = xp.createMenu("Begin Data Collection", xp.findPluginsMenu(), idx, self._menu_cb)
        xp.appendMenuItem(self.menu_id, "Begin Data Collection", "go")
        xp.log("[DC] Plugin enabled.")
        return 1



    def XPluginDisable(self):
        if self.menu_id:
            xp.destroyMenu(self.menu_id)
            self.menu_id = None



    def XPluginStop(self):
        pass



    # ============================================================
    # MESSAGES
    # ============================================================
    def XPluginReceiveMessage(self, fromWho, message, param):
        # --- If the scene is loading, wait to start a run until the scene has loaded ---#
        if message == 103 and self._placement_pending:  # XPLM_MSG_AIRPORT_LOADED
            self._placement_pending = False
            xp.registerFlightLoopCallback(self._place_on_approach, interval=1.0)



    # ============================================================
    # MENU BUTTON
    # ============================================================
    def _menu_cb(self, menuRef, itemRef):
        if itemRef == "go":
            xp.log("[DC] === Starting data collection ===")
            with suppress(Exception): xp.unregisterFlightLoopCallback(self._flare_loop)
            # with suppress(Exception): xp.unregisterFlightLoopCallback(self._monitor_loop)
            xp.setDatai(self.dr["override_joystick"], 0)
            self._placement_pending = True
            xp.placeUserAtAirport(AIRPORT)


    
    # ============================================================
    # DESCENT FLARE DETECTOR
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
        if agl > 50:
            # Simple speed control: if too fast reduce throttle, if too slow increase
            speed_error = ias - DESCENT_SPEED
            throttle = 0.5 - (speed_error / 50.0)
            throttle = max(0.0, min(1.0, throttle))
            xp.setDataf(self.dr["thro_cmd"], throttle)
        
        # --- FLARE: below 50ft, take over from AP ---
        if agl < 50:
            xp.setDatai(self.dr["override_joystick"], 1)
            
            # Hold wings level
            phi = xp.getDataf(self.dr["phi"])
            xp.setDataf(self.dr["yoke_roll"], max(-1.0, min(1.0, -phi / 25.0)))
            
            # Hold heading
            psi = xp.getDataf(self.dr["psi"])
            heading_error = HEADING - psi
            while heading_error > 180: heading_error -= 360
            while heading_error < -180: heading_error += 360
            xp.setDataf(self.dr["yoke_heading"], max(-1.0, min(1.0, heading_error / 20.0)))
            
            if agl > 15:
                xp.setDataf(self.dr["yoke_pitch"], 0.05)
                xp.setDataf(self.dr["thro_cmd"], 0.15)
            elif agl > 5:
                xp.setDataf(self.dr["yoke_pitch"], 0.15)
                xp.setDataf(self.dr["thro_cmd"], 0.0)
            else:
                xp.setDataf(self.dr["yoke_pitch"], 0.2)
                xp.setDataf(self.dr["thro_cmd"], 0.0)
        
        # Touchdown
        if agl < 5 or xp.getDatai(self.dr["gnd"]):
            xp.setDatai(self.dr["override_joystick"], 0)
            xp.setDataf(self.dr["thro_cmd"], 0.0)
            xp.log(f"[AP] Touchdown! IAS={ias:.0f}")
            xp.commandOnce(self.cmd["toggle_movie"])
            return 0
        
        return -1



    def _get_runway_polygon(self):
        """
        Project runway corners to screen pixels. Returns list of (x_norm, y_norm) 
        normalized 0-1, or None if runway not visible.
        """
        # Read matrices
        world_mat = [0.0] * 16
        proj_mat = [0.0] * 16
        xp.getDatavf(self.dr["world_matrix"], world_mat, 0, 16)
        xp.getDatavf(self.dr["proj_matrix"], proj_mat, 0, 16)
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
            px, py, behind, clip = project_local_to_pixel(lx, ly, lz, world_mat, proj_mat, sw, sh)
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



    def _monitor_loop(self, since_last, elapsed, counter, refcon):
        if not hasattr(self, '_mon_accum'):
            self._mon_accum = 0.0
        self._mon_accum += since_last
        
        if True: #self._mon_accum >= 0.05:
            self._mon_accum = 0.0
            agl = xp.getDataf(self.dr["agl"])
            ap_mode = xp.getDatai(self.dr["ap"])
            ap_state = xp.getDatai(self.dr["ap_state"])
            heading = xp.getDataf(self.dr["psi"])
            pitch = xp.getDataf(self.dr["theta"])
            speed = xp.getDataf(self.dr["ias"])
            sim_speed_actual = xp.getDataf(self.dr["sim_speed_actual"])

            poly = self._get_runway_polygon()
            label_path = os.path.join(self._label_dir, f"{self._frame_idx:06d}.txt")
            if poly:
                # YOLO format: class_id x1 y1 x2 y2 ... xn yn
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                label_line = f"0 {coords}\n"  # class 0 = runway
                with open(label_path, "w") as f:
                    f.write(label_line)
            else:
                open(label_path, "w").close()

            self._frame_idx += 1
            xp.log(f"[MON] AGL={agl:.0f} AP={ap_mode}")
        
        if agl < 5 or xp.getDatai(self.dr["gnd"]):
            return 0
        
        return 0.05



    def _randomize_wind(self):
        # Take control of weather from X-Plane
        xp.setDatai(self.dr["weather_source"], 1)  # 1 = manual/plugin
        xp.setDatai(self.dr["update_immediately"], 1)
    
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
    
        xp.commandOnce(self.cmd["regen_weather"])
        xp.log(f"[DC] Wind: {base_speed:.1f} m/s from {base_dir:.0f}°, turb={turb:.2f}")



    def _randomize_visibility(self):
        if random.random() < CLEAR_CHANCE:
            vis = 15.0
        else:
            vis = random.uniform(MIN_VIS, 15.0)
        xp.setDataf(self.dr["visibility"], vis)
        xp.log(f"[DC] Visibility: {vis:.1f} SM")



    def _set_clouds(self):
        xp.setDatavf(self.dr["cloud_base"], CLOUD_BASES, 0, NUM_CLOUD_LAYERS)
        xp.setDatavf(self.dr["cloud_tops"], CLOUD_TOPS, 0, NUM_CLOUD_LAYERS)
        xp.setDatavf(self.dr["cloud_coverage"], CLOUD_COVERAGE, 0, NUM_CLOUD_LAYERS)
        xp.log(f"[DC] Clouds: base={CLOUD_BASES[0]:.0f}m cov={CLOUD_COVERAGE[0]:.0%}")



    def _randomize_precipitation(self):
        # Scale rain to lowest cloud layer's coverage
        coverages = [0.0] * 3
        xp.getDatavf(self.dr["cloud_coverage"], coverages, 0, 3)
        max_coverage = max(coverages)
        if max_coverage < 0.3:
            rain = 0.0  # no rain with sparse clouds
        else:
            if random.random() < RAIN_CHANCE:
                rain = random.uniform(0.0, MAX_RAIN) * (max_coverage / 0.5)
            else:
                rain = 0.0
        xp.setDataf(self.dr["rain_pct"], rain)
        xp.log(f"[DC] Rain: {rain:.2f} (cloud coverage: {max_coverage:.2f})")



    def _randomize_time(self):
        # Disable system time so our writes stick
        xp.setDatai(self.dr["use_sys_time"], 0)
    
        UTC_OFFSET = TIMEZONE_OFFSET
    
        if random.random() < DAYTIME_CHANCE:
            # Daytime: 8am - 5pm local
            local_sec = random.uniform(28800, 61200)
        else:
            # Nighttime: 8pm - 5am local
            if random.random() < 0.5:
                local_sec = random.uniform(72000, 86400)
            else:
                local_sec = random.uniform(0, 18000)
    
        zulu_sec = (local_sec - UTC_OFFSET * 3600) % 86400
        xp.setDataf(self.dr["zulu_time"], zulu_sec)
    
        hours = int(local_sec // 3600)
        mins = int((local_sec % 3600) // 60)
        xp.log(f"[DC] Time: {hours:02d}:{mins:02d} local ({'day' if 28800 <= local_sec <= 61200 else 'night'})")



    def _place_on_approach(self, since_last, elapsed, counter, refcon):

        self._placement_pending = False

        self._frame_idx = 0
        self._label_dir = "/home/linok/Downloads/test_labels"
        os.makedirs(self._label_dir, exist_ok=True)

        # Randomize conditions (except cloud cover)
        self._randomize_wind()
        self._randomize_visibility()
        self._set_clouds()
        self._randomize_precipitation()
        self._randomize_time()

        # Convert both points to X-Plane local coords
        sx, sy, sz = xp.worldToLocal(START_LAT, START_LONG, START_ALT)
        tx, ty, tz = xp.worldToLocal(R1_LAT, R1_LONG, ELEV)

        # Compute true heading from start to threshold using local coords
        # X-Plane: +X=east, +Y=up, +Z=south
        dx = tx - sx
        dy = ty - sy
        dz = tz - sz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        true_heading = math.degrees(math.atan2(dx, -dz)) % 360.0
        horiz_dist = math.sqrt(dx*dx + dz*dz)
        pitch = -math.degrees(math.atan2(-dy, horiz_dist))

        xp.log(f"[DC] True heading: {true_heading:.1f}, pitch: {pitch:.1f}")
        xp.log(f"[DC] Threshold local: ({tx:.0f}, {ty:.0f}, {tz:.0f})")

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

        ap_state = xp.getDatai(self.dr["ap_state"])
        xp.log(f"[DC] Start: ({sx:.0f}, {sy:.0f}, {sz:.0f})")
        xp.log(f"[DC] AP state: {ap_state} ({bin(ap_state)})")
        xp.log("[DC] === Done ===")
        
        agl = xp.getDataf(self.dr["agl"])
        ap_mode = xp.getDatai(self.dr["ap"])
        
        # === FLARE LOOP === #
        self._flare_startup = 0.0
        xp.registerFlightLoopCallback(self._flare_loop, interval=-1)
        # xp.registerFlightLoopCallback(self._monitor_loop, interval=-1)

        # === SPEEEEEEED === #
        # xp.setDatai(self.dr["sim_speed"], 3)

        # === RECORDING === #
        xp.commandOnce(self.cmd["toggle_movie"])

        return 0
