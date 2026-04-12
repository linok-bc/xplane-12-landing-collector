"""
Place in: <X-Plane 12>/Resources/plugins/PythonPlugins/

Places the aircraft on final approach toward KSFO 28R.
Computes heading and velocity from hardcoded start/end points.
"""

import os
import time
import math
import random
from XPPython3 import xp

# ============================================================================
# CONFIGURATION
# ============================================================================

# === AIRPORT ===
AIRPORT = "KSFO"
# these should not be writen directly, they're completely dependent on AIRPORT
RUNWAY_LENGTH_M = 3048.0  # KSFO 28R length in meters
RUNWAY_WIDTH_M = 61.0     # runway width in meters
RUNWAY_CORNERS = None 
# THRESHOLD_LAT, THRESHOLD_LON = 37.6166072,-122.3670059
THRESHOLD_LAT, THRESHOLD_LON = 37.6135340, -122.3571551
THRESHOLD_HEAD = 298
THRESHOLD_ELEV_M = 3.9624
ILS_FREQ = 11170  # 111.70 MHz (I-GWQ)

# === WIND ===
MAX_WIND_SPEED =    3.0 # m/s
MAX_GUST_EXTRA =    0.5 # m/s
MAX_TURBULENCE =    0.15

# === VISIBILITY ===
MIN_VISIBILITY =    10.0 # less than 15!
NONCLEAR_CHANCE =   0.25

# === RAIN ===
MAX_RAIN =          1.0
RAIN_CHANCE =       0.5

# === TIME OF DAY ===
DAYTIME_CHANCE =    0.9
TIMEZONE_OFFSET =   -8  # should never be directly written; will depend on AIRPORT


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def degrees_to_rads(deg: float):
    return deg * (math.pi / 180)

def degrees_to_meters(nm: float):
    return nm * 60 * 1852

def get_start_position(
        rwy_lat: float,
        rwy_lon: float,
        rwy_head: float,
        rwy_elev: float,
        start_dist: float,
        start_gamma
    ):
    """
    [in]
        rwy_lat (float): runway latitude in degrees (Earth distance)
        rwy_lon (float): runway longitude in degrees (Earth distance)
        rwy_head (float): runway orientation in degrees (angles)
        rwy_elev (float): runway elevation above sea level in meters
        start_dist (float): distance to start from in nautical miles
        start_gamma (float): approach angle

    [out]
        start_lat (float): starting latitude in degrees
        start_lon (float): starting longitude in degrees
        start_alt (float): starting altitude of the plane in meters
    """
    # convert start_dist to degrees
    start_dist = start_dist / 60

    # apply offset to rwy_lat/long based on heading
    start_lat = rwy_lat - start_dist * math.cos(degrees_to_rads(rwy_head))
    start_lon = rwy_lon - start_dist * math.sin(degrees_to_rads(rwy_head)) / math.cos(degrees_to_rads(rwy_lat))
    
    # calculate starting angle
    start_alt = rwy_elev + degrees_to_meters(start_dist) * math.tan(degrees_to_rads(start_gamma + 0.5))

    return start_lat, start_lon, start_alt

def euler_to_quaternion(psi_deg, theta_deg, phi_deg):
    h = math.radians(psi_deg) / 2.0
    p = math.radians(theta_deg) / 2.0
    r = math.radians(phi_deg) / 2.0
    ch, sh = math.cos(h), math.sin(h)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return [
        cr * cp * ch + sr * sp * sh,
        sr * cp * ch - cr * sp * sh,
        cr * sp * ch + sr * cp * sh,
        cr * cp * sh - sr * sp * ch,
    ]

def mat4_mul_vec4(m, v):
    """Multiply a 16-float column-major OpenGL matrix by a 4-element vector."""
    return [
        m[0]*v[0] + m[4]*v[1] + m[8]*v[2]  + m[12]*v[3],
        m[1]*v[0] + m[5]*v[1] + m[9]*v[2]  + m[13]*v[3],
        m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3],
        m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3],
    ]

def project_local_to_pixel(lx, ly, lz, world_mat, proj_mat, screen_w, screen_h):
    eye = mat4_mul_vec4(world_mat, [lx, ly, lz, 1.0])
    clip = mat4_mul_vec4(proj_mat, eye)

    if clip[3] <= 0.0:
        return (clip[0], clip[1], True, clip)

    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]

    px = (ndc_x + 1.0) * 0.5 * screen_w
    py = (ndc_y + 1.0) * 0.5 * screen_h
    return (px, py, False, clip)

def get_runway_points(threshold_lat, threshold_lon, threshold_elev, heading, length_m, width_m, n=20):
    hdg_rad = math.radians(heading)
    half_w = width_m / 2.0
    perp_rad = hdg_rad + math.pi / 2.0

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(threshold_lat))

    dx_cross = half_w * math.sin(perp_rad)
    dy_cross = half_w * math.cos(perp_rad)

    left = []
    right = []
    for i in range(n + 1):
        frac = i / n
        along = length_m * frac
        lat = threshold_lat + along * math.cos(hdg_rad) / m_per_deg_lat
        lon = threshold_lon + along * math.sin(hdg_rad) / m_per_deg_lon
        left.append((lat - dy_cross / m_per_deg_lat, lon - dx_cross / m_per_deg_lon, threshold_elev))
        right.append((lat + dy_cross / m_per_deg_lat, lon + dx_cross / m_per_deg_lon, threshold_elev))

    return left + list(reversed(right))


# === APPROACH ===
START_DIST = 5.0
START_GAMMA = 3.0
START_LAT, START_LON, START_ALT_M = get_start_position(
        THRESHOLD_LAT, 
        THRESHOLD_LON,
        THRESHOLD_HEAD,
        THRESHOLD_ELEV_M, 
        START_DIST, 
        START_GAMMA,
    )

APPROACH_SPEED_KTS = 80.0

RUNWAY_CORNERS = get_runway_points(
    THRESHOLD_LAT, THRESHOLD_LON, THRESHOLD_ELEV_M,
    THRESHOLD_HEAD, RUNWAY_LENGTH_M, RUNWAY_WIDTH_M, n=20
)

def clip_polygon_to_screen(vertices, screen_w, screen_h):
    """
    Sutherland-Hodgman clipping of a polygon to the screen rectangle.
    vertices: list of (x, y) tuples.
    Returns clipped list of (x, y) tuples.
    """
    def clip_edge(poly, x0, y0, x1, y1):
        """Clip polygon against one edge defined by line from (x0,y0) to (x1,y1). 
           Points to the LEFT of the edge (interior) are kept."""
        if not poly:
            return []
        result = []
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]
            curr_side = (x1 - x0) * (curr[1] - y0) - (y1 - y0) * (curr[0] - x0)
            prev_side = (x1 - x0) * (prev[1] - y0) - (y1 - y0) * (prev[0] - x0)
            if curr_side >= 0:
                if prev_side < 0:
                    result.append(_intersect(prev, curr, x0, y0, x1, y1))
                result.append(curr)
            elif prev_side >= 0:
                result.append(_intersect(prev, curr, x0, y0, x1, y1))
        return result

    def _intersect(p1, p2, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        dp_x, dp_y = p2[0] - p1[0], p2[1] - p1[1]
        denom = dp_x * dy - dp_y * dx
        if abs(denom) < 1e-10:
            return p2
        t = ((x0 - p1[0]) * dy - (y0 - p1[1]) * dx) / denom
        return (p1[0] + t * dp_x, p1[1] + t * dp_y)

    # Clip against all 4 screen edges (defined counter-clockwise)
    poly = list(vertices)
    poly = clip_edge(poly, 0, 0, screen_w, 0)          # bottom
    poly = clip_edge(poly, screen_w, 0, screen_w, screen_h)  # right
    poly = clip_edge(poly, screen_w, screen_h, 0, screen_h)  # top
    poly = clip_edge(poly, 0, screen_h, 0, 0)          # left
    return poly

def clip_near_plane(clips, screen_pts, behind_flags, sw, sh):
    NEAR_EPS = 0.01
    n = len(screen_pts)
    result = []
    for i in range(n):
        j = (i + 1) % n
        if not behind_flags[i]:
            result.append(screen_pts[i])
        if behind_flags[i] != behind_flags[j]:
            w_i = clips[i][3]
            w_j = clips[j][3]
            t = (NEAR_EPS - w_i) / (w_j - w_i)
            t = max(0.0, min(1.0, t))
            interp = [clips[i][k] + t * (clips[j][k] - clips[i][k]) for k in range(4)]
            if interp[3] > 0:
                ndc_x = interp[0] / interp[3]
                ndc_y = interp[1] / interp[3]
                result.append(((ndc_x + 1.0) * 0.5 * sw, (ndc_y + 1.0) * 0.5 * sh))
    return result


class PythonInterface:

    def __init__(self):
        self.dr = {}
        self.cmd = {}
        self.menu_id = None 
        self._placement_pending = False

    def XPluginStart(self):
        return (
            "XPPython3 Landing Data Collector",
            "com.data_collection.landing",
            "Collect many, many runway landing sequences"
        )

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
            "Q_rate":               "sim/flightmodel/position/Q",
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

            # visibility
            "visibility":           "sim/weather/region/visibility_reported_sm",

            # cloud cover
            "cloud_base":           "sim/weather/region/cloud_base_msl_m",
            "cloud_tops":           "sim/weather/region/cloud_tops_msl_m",
            "cloud_coverage":       "sim/weather/region/cloud_coverage_percent",

            # rain
            "rain_pct":             "sim/weather/region/rain_percent",
            
            # time of day; need to shift from UTC
            "zulu_time":        "sim/time/zulu_time_sec",
            "use_sys_time":     "sim/time/use_system_time",

            # world to camera transformations
            "world_matrix":     "sim/graphics/view/world_matrix",
            "proj_matrix":      "sim/graphics/view/projection_matrix_3d",
            "screen_w":         "sim/graphics/view/window_width",
            "screen_h":         "sim/graphics/view/window_height",

            # speed up the sim for data collection purposes
            "sim_speed":            "sim/time/sim_speed",
            "sim_speed_actual" :    "sim/time/sim_speed_actual"
        }
        for key, path in refs.items():
            self.dr[key] = xp.findDataRef(path)
            if self.dr[key] is None:
                xp.log(f"[AP] WARN: dataref not found: {path}")

        # register relevant commands
        cmds = {
            # for autopilot
            "servos":   "sim/autopilot/servos_on",
            "approach": "sim/autopilot/approach",

            # weather control
            "regen_weather":    "sim/operation/regen_weather",

            # screen capture
            "toggle_movie": "sim/operation/video_record_toggle",
        }
        self.cmd = {}
        for key, path in cmds.items():
            self.cmd[key] = xp.findCommand(path)

        # make interactable button in X-Plane
        idx = xp.appendMenuItem(xp.findPluginsMenu(), "Begin Data Collection", 0)
        self.menu_id = xp.createMenu(
            "Begin Data Collection", xp.findPluginsMenu(), idx, self._menu_cb
        )
        xp.appendMenuItem(self.menu_id, "Begin Data Colletion", "go")
        xp.log("[DC] Plugin enabled.")
        return 1



    def XPluginDisable(self):
        if self.menu_id:
            xp.destroyMenu(self.menu_id)
            self.menu_id = None



    def XPluginStop(self):
        pass



    def XPluginReceiveMessage(self, fromWho, message, param):
        # --- If the scene is loading, wait to start a run until the scene has loaded ---#
        if message == 103 and self._placement_pending:  # XPLM_MSG_AIRPORT_LOADED
            self._placement_pending = False
            xp.registerFlightLoopCallback(self._place_on_approach, interval=1.0)



    def _menu_cb(self, menuRef, itemRef):
        if itemRef == "go":
            xp.log("[DC] === Starting data collection ===")
            try:
                xp.unregisterFlightLoopCallback(self._flare_loop)
            except:
                pass
            try:
                xp.unregisterFlightLoopCallback(self._monitor_loop)
            except:
                pass
            xp.setDatai(self.dr["override_joystick"], 0)
            self._placement_pending = True
            xp.placeUserAtAirport(AIRPORT)



    def _draw_cb(self, phase, is_before, refcon):
        poly = self._get_runway_polygon()
        label_path = os.path.join(self._label_dir, f"{self._frame_idx:06d}.txt")
        
        if poly:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            with open(label_path, "w") as f:
                f.write(f"0 {coords}\n")
        else:
            # Empty file = no runway visible
            open(label_path, "w").close()
        
        self._frame_idx += 1
        return 1



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
            speed_error = ias - APPROACH_SPEED_KTS
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
            heading_error = THRESHOLD_HEAD - psi
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
        for lat, lon, elev in RUNWAY_CORNERS:
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
            xp.log(f"[MON] AGL={agl:.0f} AP={ap_mode} coords={label_line}")
        
        if agl < 5 or xp.getDatai(self.dr["gnd"]):
            return 0
        
        return 0.05



    def _randomize_wind(self):
        # Take control of weather from X-Plane
        xp.setDatai(self.dr["weather_source"], 1)  # 1 = manual/plugin
        xp.setDatai(self.dr["update_immediately"], 1)
    
        # Random surface wind: 0-15 kts -> 0-7.7 m/s
        base_speed = random.uniform(0.0, MAX_WIND_SPEED)
        base_dir = random.uniform(0.0, 360.0)
        gust_extra = random.uniform(0.0, MAX_GUST_EXTRA)  # up to ~5 kts gust on top
        turb = random.uniform(0.0, MAX_TURBULENCE)
    
        # Fill all layers with similar values (slight increase with altitude)
        speeds = [base_speed + i * 0.3 for i in range(13)]
        dirs = [base_dir for _ in range(13)]
        turbs = [turb for _ in range(13)]
    
        xp.setDatavf(self.dr["wind_speed"], speeds, 0, 13)
        xp.setDatavf(self.dr["wind_direction"], dirs, 0, 13)
        xp.setDatavf(self.dr["wind_turbulence"], turbs, 0, 13)
    
        xp.commandOnce(self.cmd["regen_weather"])
        xp.log(f"[DC] Wind: {base_speed:.1f} m/s from {base_dir:.0f}°, turb={turb:.2f}")



    def _randomize_visibility(self):
        if random.random() > NONCLEAR_CHANCE:
            vis = 15.0
        else:
            vis = random.uniform(MIN_VISIBILITY, 15.0)
        xp.setDataf(self.dr["visibility"], vis)
        xp.log(f"[DC] Visibility: {vis:.1f} SM")



    def _randomize_clouds(self):
        n_layers = 3

        bases = []
        tops = []
        coverages = []
        for i in range(n_layers):
            if i == 0:
                # Main layer: 600-3000m MSL, 0-50% coverage
                base = random.uniform(600.0, 3000.0)
                coverage = random.uniform(0.0, 0.5)
            else:
                # Upper layers: higher, thinner
                base = random.uniform(3000.0 + i * 1000, 6000.0 + i * 1000)
                coverage = random.uniform(0.0, 0.2)
            top = base + random.uniform(100.0, 500.0)
            bases.append(base)
            tops.append(top)
            coverages.append(coverage)

        xp.setDatavf(self.dr["cloud_base"], bases, 0, n_layers)
        xp.setDatavf(self.dr["cloud_tops"], tops, 0, n_layers)
        xp.setDatavf(self.dr["cloud_coverage"], coverages, 0, n_layers)
        xp.log(f"[DC] Clouds: base={bases[0]:.0f}m cov={coverages[0]:.0%}")



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
    
        UTC_OFFSET = -7  # KSFO PDT; change to -8 for PST
    
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

        # Randomize conditions
        self._randomize_wind()
        self._randomize_visibility()
        self._randomize_clouds()
        self._randomize_precipitation()
        self._randomize_time()

        # Convert both points to X-Plane local coords
        sx, sy, sz = xp.worldToLocal(START_LAT, START_LON, START_ALT_M)
        tx, ty, tz = xp.worldToLocal(THRESHOLD_LAT, THRESHOLD_LON, THRESHOLD_ELEV_M)

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
        speed_ms = APPROACH_SPEED_KTS * 0.5144
        xp.setDataf(self.dr["vx"], speed_ms * dx / dist)
        xp.setDataf(self.dr["vy"], speed_ms * dy / dist)
        xp.setDataf(self.dr["vz"], speed_ms * dz / dist)
        xp.setDataf(self.dr["P"], 0.0)
        xp.setDataf(self.dr["Q_rate"], 0.0)
        xp.setDataf(self.dr["R"], 0.0)

        # === THROTTLE ===
        xp.setDataf(self.dr["thro_cmd"], 0.7)

        # === RADIOS ===
        xp.setDatai(self.dr["hsi_src"], 0)
        xp.setDatai(self.dr["nav1"], ILS_FREQ)
        xp.setDataf(self.dr["nav1_obs"], THRESHOLD_HEAD)

        # === AUTOPILOT ===
        xp.setDataf(self.dr["ap_hdg"], THRESHOLD_HEAD)
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
        xp.registerFlightLoopCallback(self._monitor_loop, interval=-1)

        # === CAMERA CONTROL === #
        # xp.controlCamera(xp.ControlCameraUntilViewChanges, self._camera_cb)

        # === SPEEEEEEED === #
        xp.setDatai(self.dr["sim_speed"], 3)

        # === RECORDING === #
        xp.commandOnce(self.cmd["toggle_movie"])

        return 0
