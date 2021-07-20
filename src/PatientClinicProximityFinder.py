import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import os.path
from functools import partial
import json
import logging.config
import yaml
import pygtrie
import requests
from retry import retry
from geolib import geohash
from geopy.exc import GeocoderServiceError, GeopyError
import math


with open(r"./../config/logging_config.yml", 'r') as config:
    config = yaml.safe_load(config)
    logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

with open("./../config/app_config.json", "r") as config:
    app_config = json.load(config)
    country_codes = app_config["country_code"]
    retry_count = int(app_config["retry_count"])
    delay = int(app_config["delay"])
    backoff = int(app_config["backoff"])
    # These two setting are used to generate proximity geo hashes within a specific radius and specific precision
    # proximity radius is in meters
    proximity_radius = float(app_config["proximity_radius"])
    # No of digits in the generated proximity hash
    geohash_precision = int(app_config["geohash_precision"])


def in_circle_check(latitude, longitude, centre_lat, centre_lon, radius):
    x_diff = longitude - centre_lon
    y_diff = latitude - centre_lat
    if math.pow(x_diff, 2) + math.pow(y_diff, 2) <= math.pow(radius, 2):
        return True
    return False


def get_centroid(latitude, longitude, height, width):
    y_cen = latitude + (height / 2)
    x_cen = longitude + (width / 2)
    return x_cen, y_cen


def convert_to_latlon(y, x, latitude, longitude):
    pi = 3.14159265359
    r_earth = 6371000
    lat_diff = (y / r_earth) * (180 / pi)
    lon_diff = (x / r_earth) * (180 / pi) / math.cos(latitude * pi/180)
    final_lat = latitude+lat_diff
    final_lon = longitude+lon_diff
    return final_lat, final_lon


def create_geohash(latitude, longitude, radius, precision):
    """
    Generates a list of geohashes within the specified radius (in meters) of a geolocation with
    specified precision
    :param latitude: location latitude
    :param longitude: location longitude
    :param radius: proximity radius in meters
    :param precision: No of digits in the generated proximity hash
    :return: comma separated string of geohashes
    """
    x = 0.0
    y = 0.0
    points = []
    geohashes = []
    grid_width = [5009400.0, 1252300.0, 156500.0, 39100.0, 4900.0, 1200.0, 152.9, 38.2, 4.8, 1.2, 0.149, 0.0370]
    grid_height = [4992600.0, 624100.0, 156000.0, 19500.0, 4900.0, 609.4, 152.4, 19.0, 4.8, 0.595, 0.149, 0.0199]
    height = (grid_height[precision - 1])/2
    width = (grid_width[precision-1])/2
    lat_moves = int(math.ceil(radius / height)) #4
    lon_moves = int(math.ceil(radius / width)) #2

    for i in range(0, lat_moves):
        temp_lat = y + height*i
        for j in range(0,lon_moves):
            temp_lon = x + width*j
            if in_circle_check(temp_lat, temp_lon, y, x, radius):
                x_cen, y_cen = get_centroid(temp_lat, temp_lon, height, width)
                lat, lon = convert_to_latlon(y_cen, x_cen, latitude, longitude)
                points += [[lat, lon]]
                lat, lon = convert_to_latlon(-y_cen, x_cen, latitude, longitude)
                points += [[lat, lon]]
                lat, lon = convert_to_latlon(y_cen, -x_cen, latitude, longitude)
                points += [[lat, lon]]
                lat, lon = convert_to_latlon(-y_cen, -x_cen, latitude, longitude)
                points += [[lat, lon]]

    for point in points:
        geohashes += [geohash.encode(point[0], point[1], precision)]

    return ','.join(set(geohashes))


# Twp way dictionary mapping column names in csv (or dataframe) to search criteria column names
# and vice versa (for nominatim API)
def get_pat_col_searchcol_map():
    pat_map = {"Postal Code": "postalcode",
               "FSA": "postalcode",
               "postalcode": "Postal Code",
               "Address": "street",
               "street": "Address",
               "City": "city",
               "city": "City",
               "Province": "state",
               "state": "Province"}
    return pat_map


def get_clinic_col_searchcol_map():
    clinic_map = {"Postal Code": "postalcode",
                  "FSA": "postalcode",
                  "postalcode": "Postal Code",
                  "Clinic Address": "street",
                  "street": "Clinic Address",
                  "Clinic City": "city",
                  "city": "Clinic City",
                  "Province": "state",
                  "state": "Province"}
    return clinic_map


def address_postalcode_search(geocode, search_params, initial_call=False):
    """
    Makes a call to Nominatim api using address and postal code as search criteria
    :param geocode: geopy RateLimiter object to do throttled search using Nominatim api
    :param search_params: Dictionary with search criteria
    :param initial_call: bool value indicating if this is the first call to the method
    :return: geopy.location.Location object or None if no search result found
    """
    address = search_params["street"]
    postal_code = search_params["postalcode"] if "postalcode" in search_params.keys() else ""
    if not initial_call:
        if not address:
            return None
        # remove the last word from address
        address_tokens = address.split(" ")[0:-1]
        address = " ".join(address_tokens) if len(address_tokens) > 0 else ""
        search_params["street"] = address
    if postal_code:
        search_params.pop("postalcode")
    locations = geocode(search_params)
    search_params["postalcode"] = postal_code
    if locations is not None and len(locations) > 0 and postal_code:
        # if address or one of its substrings returns a location, check if adding postalcode search criteria
        # refines the search
        refined_locations = geocode(search_params)
        if refined_locations is not None and len(refined_locations) > 0:
            return refined_locations[0]
        else:
            return locations[0]
    if locations is None and search_params["street"]:
        return address_postalcode_search(geocode, search_params)
    else:
        return None


def no_address_search(geocode, search_params):
    """
    This method is called when address and postal code combination yields no search results.
    The search criteria used are postal_code and fsa combination followed by only fsa. As last
    resort only fixed search criteria of province and city is used.
    :param geocode: geopy RateLimiter object to do throttled search using Nominatim api
    :param search_params: Dictionary with search criteria
    :return: geopy.location.Location object or None if no search result found
    """
    locations = geocode(search_params)
    if locations is None:
        return None
    elif len(locations) > 0:
        return locations[0]


@retry((GeopyError, GeocoderServiceError), tries=retry_count, delay=delay, backoff=backoff, logger=logger)
def get_geocode(type, geocode, df_row, fixed_search_cols, var_search_cols=[], initialCall=False):
    """
    Gets the geocode for either a clinic or a patient record
    :param type: string "patient" or "clinic"
    :param geocode: geopy RateLimiter object to do throttled search using Nominatim api
    :param df_row: dataframe row (from patient or clinic dataframe)
    :param fixed_search_cols: list of search columns used as search criteria in all searches (Province and City)
    :param var_search_cols: list of lists containing column names for more precise search
    :param initialCall: bool value indicating if this is the first call to the method
    :return: pandas series object containing 'Geo_Cols', 'Geo_Code' and 'Geo_Hash'
    """
    var_search_cols = get_var_search_cols(initialCall, type, var_search_cols)
    prefix = 'Pat_' if type.lower() == 'patient' else 'Clinic_'
    id_col = "ID" if type.lower() == 'patient' else 'Clinic ID'
    col_searchcol_map = get_pat_col_searchcol_map() if type.lower() == 'patient' else get_clinic_col_searchcol_map()
    id = df_row[id_col]
    col_labels = [prefix+x for x in ['Geo_Cols', 'Geo_Code', 'Geo_Hash']]
    search_params = get_search_params(df_row, fixed_search_cols, var_search_cols, col_searchcol_map, type, id)
    if "street" in search_params.keys():
        location = address_postalcode_search(geocode, search_params, True)
    else:
        location = no_address_search(geocode, search_params)
    if location is not None:
        search_cols = ",".join(["FSA" if "FSA" in var_search_cols else col_searchcol_map[key]
                                for key in search_params.keys()])
        lat = location.latitude
        lon = location.longitude
        logger.info(f"For {type}ID: {id} with {search_params} => latitude = {lat}, longitude = {lon}")
        loc_geohash = geohash.encode(lat, lon, 12)
        return pd.Series([search_cols, (lat, lon), loc_geohash], index=col_labels)
    elif len(var_search_cols) > 0 and location is None:
        # Remove the most precise search criteria which is at the top of search criteria stack
        var_search_cols.pop()
        return get_geocode(type, geocode, df_row, fixed_search_cols, var_search_cols=var_search_cols)
    else:
        # Neither variable nor fixed search criteria yield a geocode
        return pd.Series([None, (None, None), None], index=col_labels)


def get_var_search_cols(initialCall, type, var_search_cols):
    """
    Initialize a list of lists containing column names for more precise search
    :param initialCall: bool value indicating if this is the first call to the method
    :param type: string "patient" or "clinic"
    :param var_search_cols: list of lists containing column names for more precise search
    :return: list of lists containing column names for more precise search
    """
    if initialCall and type.lower() == 'clinic':
        var_search_cols = [["FSA"],
                           ["Postal Code"],
                           ["Postal Code", "Clinic Address"]]
    elif initialCall and type.lower() == 'patient':
        var_search_cols = [["FSA"],
                           ["Postal Code"],
                           ["Postal Code", "Address"]]
    return var_search_cols


def get_search_params(df_row, fixed_search_cols, var_search_cols, col_searchcol_map, type, id):
    """
    Creates a dictionary of search parameters using both fixed and precise search criteria
    :param df_row: dataframe row
    :param fixed_search_cols: list of search columns used as search criteria in all searches (Province and City)
    :param var_search_cols: list of lists containing column names for more precise search
    :param col_searchcol_map: Dictionary mapping column names in csv (or dataframe) to search criteria column names
     and vice versa (for nominatim API)
    :param type: string "patient" or "clinic"
    :param id: id of the dataframe row
    :return: dictionary with search parameters
    """
    if len(var_search_cols) == 0:
        var_search_params = []
        logger.critical(f"Exhausted all variable geocode search criteria for {type}id: {id}. No geocode found. "
                        f"Geocode will be returned on the basis of fixed search cols")
    else:
        var_search_params = [(col_searchcol_map[df_col], df_row[df_col]) for df_col in var_search_cols[-1]]
    fixed_search_params = [(col_searchcol_map[df_col], df_row[df_col]) for df_col in fixed_search_cols]
    search_params = dict(var_search_params + fixed_search_params)
    return search_params


def get_clinic_geocode(geolocator, df_clinics):
    """
    Get the geocode for a clinic record
    :param geolocator: geopy RateLimiter object to do throttled search using Nominatim api
    :param df_clinics: clinics dataframe
    :return: clinics dataframe with Clinic_Geo_Cols, Clinic_Geo_Code and Clinic_Geo_Hash columns
    """
    fixed_search_cols = ["Province", "Clinic City"]
    df_clinics_geocode = df_clinics.apply(
        lambda row: get_geocode("clinic", geolocator, row, fixed_search_cols, initialCall=True), axis=1)

    return pd.concat([df_clinics, df_clinics_geocode], axis=1)


def get_patient_geocode(geolocator, df_patients):
    """
    Get the geocode for a patient record
    :param geolocator: geopy RateLimiter object to do throttled search using Nominatim api
    :param df_patients: clinics dataframe
    :return: patients dataframe with Pat_Geo_Cols, Pat_Geo_Code and Pat_Geo_Hash columns
    """
    fixed_search_cols = ["Province", "City"]
    df_patients_geocode = df_patients.apply(
        lambda row: get_geocode("patient", geolocator, row, fixed_search_cols, initialCall=True), axis=1)

    return pd.concat([df_patients, df_patients_geocode], axis=1)


def get_geohash_nearby_clinics(pat_geohash, clinic_geohash_trie):
    """
    Perform a string prefix search to find clinic with geohashes that match a substring of patient geohash
    :param pat_geohash: one of patient's location geohash
    :param clinic_geohash_trie: Trie containing all clinic geohashes
    :return: list of matching geohashes
    """
    if not pat_geohash:
        return None
    try:
        nearest_clinics = clinic_geohash_trie.keys(pat_geohash)
    except KeyError:
        pat_geohash = pat_geohash[0:-1]
        return get_geohash_nearby_clinics(pat_geohash, clinic_geohash_trie)
    if nearest_clinics is None or len(nearest_clinics) == 0:
        pat_geohash = pat_geohash[0:-1]
        return get_geohash_nearby_clinics(pat_geohash, clinic_geohash_trie)
    else:
        return nearest_clinics


def get_pat_nearby_clinics(pat_gh, gh_to_match, clinic_gh_trie):
    """
    Perform a string prefix search to find clinic with geohashes that match a substring of
    one of the patient geohash search set
    :param pat_gh: patient's geohash
    :param gh_to_match: set containing geohashes within a proximity radius of patient's geohash as well as
    the patient's 8 immediate neighbour geohashes
    :param clinic_gh_trie: Trie containing all clinic geohashes
    :return: list of matching clinic geohashes
    """
    pat_nearby_clinics = set()
    gh_to_match.append(pat_gh)
    for gh in gh_to_match:
        gh_nearby_clinic = get_geohash_nearby_clinics(gh, clinic_gh_trie)
        if gh_nearby_clinic is not None:
            pat_nearby_clinics.update(gh_nearby_clinic)

    logger.info(f"For patient geohash = {pat_gh} nearby clinics geohashes are = {pat_nearby_clinics}")
    return pat_nearby_clinics


@retry(requests.exceptions.ConnectionError, tries=retry_count, delay=delay, backoff=backoff, logger=logger)
def get_osrm_clinic_travel_distance(df_pat_row, df_clinic_row):
    """
    Get the shortest travel distance between a patient's and a clinic's geolocation using OSRM api
    :param df_pat_row: row of patient dataframe
    :param df_clinic_row: row of clinic dataframe
    :return: travel distance in km
    """
    pat_lat, pat_lon = df_pat_row["Pat_Geo_Code"]
    clinic_lat, clinic_lon = df_clinic_row["Clinic_Geo_Code"]
    # call the OSRM API
    r = requests.get(f"http://router.project-osrm.org/route/v1/car/{pat_lon},{pat_lat};"
                     f"{clinic_lon},{clinic_lat}?overview=false""")
    # then you load the response using the json library
    # by default you get only one alternative so you access 0-th element of the `routes`
    routes = json.loads(r.content)
    fastest_route = routes.get("routes")[0]
    travel_distance = float(fastest_route["distance"] / 1000)
    logger.info(f"Travel distance between patient id = {df_pat_row['ID']} with geohash = {df_pat_row['Pat_Geo_Hash']}"
                f" and clinic id = {df_clinic_row['Clinic ID']} with geohash = {df_clinic_row['Clinic_Geo_Hash']}"
                f" is => {travel_distance}")
    return travel_distance


def get_pat_nearest_clinic(df_pat_row, df_clinics, clinic_gh_trie):
    """
    For a patient record finds the clinic with shortest travel distance
    :param df_pat_row: row of patient dataframe
    :param df_clinics: clinic dataframe
    :param clinic_gh_trie: Trie containing all clinic geohashes
    :return: pandas series object with columns in output.csv
    """
    pat_gh = df_pat_row["Pat_Geo_Hash"]
    # To handle the edge case of a when proximity searches are done near the Greenwich Meridian or the equator,
    # because in those points the MSB of x and y will be 0 or 1, so their geohashes won't share a common prefix.
    # To work around this we get the neighbours of a location's geohash and proximity search will then need to
    # find a list of geohashes that are prefixed by the original location and the neighbour's geohash
    pat_gh_neighbors = list(geohash.neighbours(pat_gh))
    pat_lat, pat_lon = df_pat_row["Pat_Geo_Code"]
    # list of geohashes that are with in a specific radius of the patient location. To handle the edge case
    # of a location being on the boundary of a geohash bounding box, we need to broaden the geohash
    # search space
    gh_in_search_radius = create_geohash(pat_lat, pat_lon, proximity_radius, geohash_precision).split(",")
    gh_to_match = pat_gh_neighbors + gh_in_search_radius
    pat_nearby_clinics = get_pat_nearby_clinics(pat_gh, gh_to_match, clinic_gh_trie)
    # Filter the clinic dataframe using nearby clinic geohashes
    df_clinics_nearby = df_clinics[df_clinics["Clinic_Geo_Hash"].isin(pat_nearby_clinics)]
    # From the nearby clinics use the shortest travel distance ( google maps ) or shortest travel time criteria
    # ( OSRM for open street maps ) to find the closest clinic
    df_clinics_nearby["Clinic_Pat_Dist"] = df_clinics_nearby.apply(
        lambda row: get_osrm_clinic_travel_distance(df_pat_row, row), axis=1
    )
    # select the clinic row with shortest travel distance
    min_distance = df_clinics_nearby["Clinic_Pat_Dist"].min()
    df_nearest_clinic = df_clinics_nearby.loc[df_clinics_nearby["Clinic_Pat_Dist"] == min_distance]
    nearest_clinic = df_nearest_clinic.iloc[0]
    # join the patient row with selected clinic row to return a series object
    result_cols = ["Patient_ID", "Pat_Geo_Cols", "Pat_Geo_Code", "Pat_Address", "Pat_Postal_Code",
                   "Pat_FSA", "Nearest_Clinic_ID", "Clinic_Geo_Cols", "Clinic_Geo_Code", "Clinic_Address",
                   "Clinic_Postal Code", "Clinic_FSA", "Clinic_Distance"]
    nearest_result = pd.Series([df_pat_row["ID"], df_pat_row["Pat_Geo_Cols"], df_pat_row["Pat_Geo_Code"],
                                df_pat_row["Address"], df_pat_row["Postal Code"], df_pat_row["FSA"],
                                nearest_clinic["Clinic ID"], nearest_clinic["Clinic_Geo_Cols"],
                                nearest_clinic["Clinic_Geo_Code"], nearest_clinic["Clinic Address"],
                                nearest_clinic["Postal Code"], nearest_clinic["FSA"],
                                nearest_clinic["Clinic_Pat_Dist"]],index=result_cols)
    return nearest_result


def get_nearest_clinics(df_patients, df_clinics):
    """
    For each patient record find the clinic with shortest travel distance
    :param df_patients: patient dataframe
    :param df_clinics: clinic dataframe
    :return: dataframe with patient and nearest clinic data
    """
    # Trie to hold the geohashes of all the clinics.
    clinic_geohash_trie = pygtrie.CharTrie()
    df_clinics_with_geohash = df_clinics[df_clinics.Clinic_Geo_Hash.notnull()]
    for clinic_gh in df_clinics_with_geohash["Clinic_Geo_Hash"]:
        clinic_geohash_trie[clinic_gh] = 1

    df_pat_nearest_clinic = df_patients[df_patients.Pat_Geo_Hash.notnull()].apply(
        lambda row: get_pat_nearest_clinic(row, df_clinics, clinic_geohash_trie), axis=1
    )
    return df_pat_nearest_clinic


def process_clinic_data(geocode):
    df_clinics = pd.read_csv('./data/clinics.csv')
    df_clinics = get_clinic_geocode(geocode, df_clinics)
    df_clinics.to_pickle("./data/clinics_with_geocode.pkl")
    return df_clinics


def process_patients_data(geocode):
    df_patients = pd.read_csv('./data/patients.csv')
    df_patients = get_patient_geocode(geocode, df_patients)
    df_patients.to_pickle("./data/patients_with_geocode.pkl")
    return df_patients


df_clinics = None
df_patients = None
geolocator = Nominatim(user_agent="test-app", timeout=5)
# Country = Canada is a fixed search criteria. We are only interested in locations in Canada
geocode_partial = partial(geolocator.geocode, exactly_one=False, country_codes=country_codes)
geocode = RateLimiter(geocode_partial, min_delay_seconds=1)

if os.path.isfile("./data/clinics_with_geocode.pkl"):
    df_clinics = pd.read_pickle("./data/clinics_with_geocode.pkl")
else:
    df_clinics = process_clinic_data(geocode)

if os.path.isfile("./data/patients_with_geocode.pkl"):
    df_patients = pd.read_pickle("./data/patients_with_geocode.pkl")
else:
    df_patients = process_patients_data(geocode)

print(df_clinics.to_string())
print("==========================================================================")
print(df_patients.to_string())
df_pat_nearest_clinic = get_nearest_clinics(df_patients, df_clinics)
print("==========================================================================")
print(df_pat_nearest_clinic.to_string())
df_pat_nearest_clinic.to_csv("./output/output.csv")
print("")

