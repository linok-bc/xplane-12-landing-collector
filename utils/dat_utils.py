def parse_apt_dat(dat_path):
    current_airport = None

    with open(dat_path) as fp:
        lines = [line.strip() for line in fp]

    out = []

    # refer to https://developer.x-plane.com/docs/specs, Airport & NAVAID tab
    for line in lines:
        line = line.split()

        # skip empty lines
        if len(line) == 0:
            continue

        # detect land airport
        if line[0] == '1':
            elev = int(line[1])
            airport_name = line[4]

        # detect runway
        elif line[0] == '100':
            runway_width = float(line[1])
            r1_name, r1_lat, r1_long = line[8:11]
            r2_name, r2_lat, r2_long = line[17:20]
            r1_lat, r1_long, r2_lat, r2_long = list(map(float, [r1_lat, r1_long, r2_lat, r2_long]))

            runway_info = {
                'elev': elev,
                'r1_name': r1_name,
                'r1_lat': r1_lat,
                'r1_long': r1_long,
                'r2_name': r2_name,
                'r2_lat': r2_lat,
                'r2_long': r2_long,
                'runway_width': runway_width
            }
            
            reverse_runway_info = {
                'elev': elev,
                'r1_name': r2_name,
                'r1_lat': r2_lat,
                'r1_long': r2_long,
                'r2_name': r1_name,
                'r2_lat': r1_lat,
                'r2_long': r1_long,
                'runway_width': runway_width
            }

            out.append([airport_name, r1_name, runway_info])
            out.append([airport_name, r2_name, reverse_runway_info])

        # irrelevant info; we ignore
        else:
            pass

    return out



def parse_ils_data(dat_path):

    with open(dat_path) as fp:
        lines = [line.strip() for line in fp]

    out = {}

    for line in lines:
        line = line.split()

        # skip empty lines
        if len(line) == 0:
            continue

        # ILS info
        if line[0] == '4':
            lat, long = float(line[1]), float(line[2])
            elev = float(line[3])
            ils_freq = int(line[4])
            # true bearing is encoded; see https://developer.x-plane.com/wp-content/uploads/2020/03/XP-NAV1150-Spec.pdf
            bearing = float(line[6]) % 360
            airport_name = line[8]
            runway_name = line[10]
            
            # formulate data into a dict
            runway_info = {
                'lat': lat,
                'long': long,
                'elev': elev,
                'ils_freq': ils_freq,
                'bearing': bearing,
            }

            # send to out
            if out.get(airport_name) == None:
                out[airport_name] = {runway_name: runway_info}
            else:
                out[airport_name][runway_name] = runway_info
            
        # irrelevant info; we ignore
        else:
            pass

    return out



def get_runways(apt_dat_path, nav_dat_path):
    runways = parse_apt_dat(apt_dat_path)
    nav_dict = parse_ils_data(nav_dat_path)
    out = [runway for runway in runways if nav_dict.get(runway[0]) and nav_dict[runway[0]].get(runway[1])]
    return out
