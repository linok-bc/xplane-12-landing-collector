import math



def degrees_to_rads(deg: float):
    """
    [in]
        degrees (float): degrees to convert to radians
    [out]
        out (float): the input expressed in terms of radians
    """
    out = deg * (math.pi / 180)
    return out



def degrees_to_meters(deg: float):
    """
    [in]
        deg (float): the degrees to turn into meters; longitudes may need to be scaled after this
    
    [out]
        out (float): the distance in meters
    """
    out = deg * 60 * 1852
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
    eye = mat4_mul_vec4(world_mat, [lx, ly, lz, 1.0])
    clip = mat4_mul_vec4(proj_mat, eye)

    if clip[3] <= 0.0:
        return (clip[0], clip[1], True, clip)

    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]

    px = (ndc_x + 1.0) * 0.5 * screen_w
    py = (ndc_y + 1.0) * 0.5 * screen_h
    return (px, py, False, clip)


def get_runway_points(lat1, lon1, lat2, lon2, width_m, n=20):
    """
    fetch the coordinates of each runway center, then convert to a polygon

    [in]
        lat1 (float): latitude of nearside runway edge
        long1 (float): longitude of nearside runway edge
        lat2 (float): latitude of farside runway edge
        long2 (float): longitude of farside runway edge
        width_m (float): the width of the runway in meters
        n (int), default=20: number of points to represent each half of the polygon
    """
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))

    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    length = math.sqrt(dx*dx + dy*dy)
    
    # Perpendicular offset
    perp_lat = -dx / length * (width_m / 2) / m_per_deg_lat
    perp_lon =  dy / length * (width_m / 2) / m_per_deg_lon

    left, right = [], []
    for i in range(n + 1):
        frac = i / n
        lat = lat1 + frac * (lat2 - lat1)
        lon = lon1 + frac * (lon2 - lon1)
        left.append((lat + perp_lat, lon + perp_lon, 0))
        right.append((lat - perp_lat, lon - perp_lon, 0))

    return left + list(reversed(right))

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
