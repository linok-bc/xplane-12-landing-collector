import math


# meters per degree of latitude (spherical approximation). Longitude is this scaled
# by cos(latitude). Used by everything in this file so that the start position and the
# runway corners are derived on the same footing.
M_PER_DEG = 111320.0



def degrees_to_rads(deg: float):
    """
    [in]
        degrees (float): degrees to convert to radians
    [out]
        out (float): the input expressed in terms of radians
    """
    out = deg * (math.pi / 180)
    return out



def get_start_position(
        rwy_lat: float,
        rwy_lon: float,
        rwy_head: float,
        rwy_elev: float,
        start_dist: float,
        start_gamma
    ):
    """
    Back off from the threshold along the reciprocal of the runway bearing.

    [in]
        rwy_lat (float): runway latitude in degrees (Earth distance)
        rwy_lon (float): runway longitude in degrees (Earth distance)
        rwy_head (float): runway orientation in degrees TRUE (angles)
        rwy_elev (float): runway elevation above sea level in meters
        start_dist (float): distance to start from, in METERS
        start_gamma (float): approach angle in degrees

    [out]
        start_lat (float): starting latitude in degrees
        start_lon (float): starting longitude in degrees
        start_alt (float): starting altitude of the plane in meters MSL
    """
    # convert start_dist to degrees
    start_dist_deg = start_dist / M_PER_DEG

    # apply offset to rwy_lat/long based on heading
    start_lat = rwy_lat - start_dist_deg * math.cos(degrees_to_rads(rwy_head))
    start_lon = rwy_lon - start_dist_deg * math.sin(degrees_to_rads(rwy_head)) / math.cos(degrees_to_rads(rwy_lat))

    # calculate starting angle. the extra half degree puts us slightly above the
    # glideslope so the autopilot captures it from above rather than chasing it up
    start_alt = rwy_elev + start_dist * math.tan(degrees_to_rads(start_gamma + 0.5))

    return start_lat, start_lon, start_alt



def euler_to_quaternion(psi_deg, theta_deg, phi_deg):
    """
    [in]
        psi_deg (float): euler angle
        theta_deg (float): euler angle
        phi_deg (float): euler angle

    [out]
        out: euler angles converted to quaternions
    """
    h = math.radians(psi_deg) / 2.0
    p = math.radians(theta_deg) / 2.0
    r = math.radians(phi_deg) / 2.0
    ch, sh = math.cos(h), math.sin(h)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    out = [
        cr * cp * ch + sr * sp * sh,
        sr * cp * ch - cr * sp * sh,
        cr * sp * ch + sr * cp * sh,
        cr * cp * sh - sr * sp * ch,
    ]
    return out



def mat4_mul_vec4(m, v):
    """Multiply a 16-float column-major OpenGL matrix by a 4-element vector."""
    return [
        m[0]*v[0] + m[4]*v[1] + m[8]*v[2]  + m[12]*v[3],
        m[1]*v[0] + m[5]*v[1] + m[9]*v[2]  + m[13]*v[3],
        m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3],
        m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3],
    ]



def project_local_to_pixel(lx, ly, lz, world_mat, proj_mat, screen_w, screen_h):
    """
    Project an OpenGL local-coordinate point to pixels.

    [out]
        px, py (float): pixel coordinates, ORIGIN BOTTOM-LEFT (OpenGL convention).
                        meaningless when behind is True
        behind (bool): True if the point is at or behind the camera plane
    """
    eye = mat4_mul_vec4(world_mat, [lx, ly, lz, 1.0])
    clip = mat4_mul_vec4(proj_mat, eye)

    if clip[3] <= 0.0:
        return (0.0, 0.0, True)

    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]

    px = (ndc_x + 1.0) * 0.5 * screen_w
    py = (ndc_y + 1.0) * 0.5 * screen_h
    return (px, py, False)



def get_runway_corners(lat1, lon1, lat2, lon2, width_m, elev_near, elev_far):
    """
    The four corners of the runway rectangle, in boundary order:

        near-left, far-left, far-right, near-right

    where 'near' is the (lat1, lon1) landing threshold and left/right are as seen
    from the cockpit on approach to that threshold.

    Each end takes its own elevation so that terrain-probed surface heights can be
    passed in; using a single published airport elevation puts the polygon several
    meters off the ground on a sloped runway, which dominates the label error on
    short final.

    [in]
        lat1, lon1 (float): landing threshold (near end)
        lat2, lon2 (float): far end of the runway
        width_m (float): the width of the runway in meters
        elev_near (float): surface elevation at the near end, meters MSL
        elev_far (float): surface elevation at the far end, meters MSL
    """
    m_per_deg_lat = M_PER_DEG
    m_per_deg_lon = M_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))

    dx = (lon2 - lon1) * m_per_deg_lon   # east component, meters
    dy = (lat2 - lat1) * m_per_deg_lat   # north component, meters
    length = math.sqrt(dx*dx + dy*dy)

    # rotating the along-track unit vector (dx, dy) by -90 degrees gives (dy, -dx),
    # which points to the RIGHT of the direction of travel
    right_lat = -dx / length * (width_m / 2) / m_per_deg_lat
    right_lon =  dy / length * (width_m / 2) / m_per_deg_lon

    near_left  = (lat1 - right_lat, lon1 - right_lon, elev_near)
    far_left   = (lat2 - right_lat, lon2 - right_lon, elev_far)
    far_right  = (lat2 + right_lat, lon2 + right_lon, elev_far)
    near_right = (lat1 + right_lat, lon1 + right_lon, elev_near)

    return [near_left, far_left, far_right, near_right]



def compute_bearing(lat1, lon1, lat2, lon2):
    """Initial true bearing from point 1 to point 2, in degrees (0=N, 90=E)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360
