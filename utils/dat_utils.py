def parse_apt_dat(dat_path):
    current_airport = None

    with open(dat_path) as fp:
        lines = [line.strip() for line in fp]

    out = []

    # refer to https://developer.x-plane.com/docs/specs, Airport & NAVAID tab
    for line in lines:
        # skip empty lines
        if len(line) == 0:
            continue

        # skip irrelevant lines
        if line[0] != '1':
            continue
        if len(line) < 3 and line[0:3] != '100':
            continue

        line = line.split()

        # detect land airport
        if line[0] == '1':
            elev = int(line[1]) * 0.3048   # feet to meters
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

    return out



def parse_ils_data(dat_path):

    with open(dat_path) as fp:
        lines = [line.strip() for line in fp]

    out = {}

    for line in lines:
        # skip empty lines
        if len(line) == 0:
            continue
        
        # irrelevant info; we ignore
        if line[0] != '4':
            continue

        # ILS info
        line = line.split()
        lat, long = float(line[1]), float(line[2])
        elev = float(line[3])
        ils = int(line[4])
        # true bearing is encoded; see https://developer.x-plane.com/wp-content/uploads/2020/03/XP-NAV1150-Spec.pdf
        heading = float(line[6]) % 360
        airport_name = line[8]
        runway_name = line[10]
        
        # formulate data into a dict
        runway_info = {
            'lat': lat,
            'long': long,
            'elev': elev,
            'ils': ils,
            'heading': heading,
        }

        # send to out
        if out.get(airport_name) == None:
            out[airport_name] = {runway_name: runway_info}
        else:
            out[airport_name][runway_name] = runway_info
            

    return out



def parse_runways_dat(dat_path):
    out = []
    with open(dat_path) as fp:
        for line in fp:
            line = line.split()
            airport_name, runway_name = line[0:2]
            elev = float(line[2])
            runway_width = float(line[3])
            r1_name, r2_name = line[4], line[7]
            r1_lat, r1_long, r2_lat, r2_long = list(map(float, line[5:7] + line[8:10]))
            ils = int(line[10])
            heading = float(line[11])
            out.append({
                'airport_name': airport_name,
                'runway_name': runway_name,
                'elev': elev,
                'runway_width': runway_width,
                'r1_name': r1_name,
                'r1_lat': r1_lat,
                'r1_long': r1_long,
                'r2_name': r2_name,
                'r2_lat': r2_lat,
                'r2_long': r2_long,
                'ils': ils,
                'heading': heading,
            })
    return out
            


def get_runways(apt_dat_path, nav_dat_path):
    runways = parse_apt_dat(apt_dat_path)
    nav_dict = parse_ils_data(nav_dat_path)
    out = []
    for runway in runways:
        if nav_dict.get(runway[0]) and nav_dict[runway[0]].get(runway[1]):
            nav = nav_dict[runway[0]][runway[1]]
            runway[2]['ils'] = nav['ils']
            runway[2]['heading'] = nav['heading']
            out.append(runway)
        else:
            pass
    return out
