###Required Libraries:
Mentioned in requirements.txt

### Running the script
1) cd path_to_PatientClinicProximityFinder/src
2) Ensure that correct clinics.csv and patients.csv file is present in src/data directory
3) Run the PatientClinicProximityFinder.py script. The script takes no parameters. After successful
completion of script, output.csv in created in src/output folder.

The script after running creates in src/data folder clinics_with_geocode.pkl and patients_with_geocode.pkl files that contain the processed input clinic and patients data including the location geocode and geohash.
In case of a clean rerun of the script again on input data, delete the .pkl files from src/data and output.csv from src/output folder. The pickle files are created as the geocoding process is time taking and once done
successfully patient and clinic pickle files can be loaded in pandas dataframe to analyze the geocoded patient and clinic data   

Note that as we are using open source Nominatim api for geocoding which has a restriction of not submitting more than
one request per second, the geocoding process on the challenge dataset of 100 clinics and 500 patients takes around 30 min to 1 hour.
The second part of the script runs once all addresses have been geocoded and the intermediary pickle files have been generated. It finds the clinic with 
shortest travel distance makes use of open source OSRM api. This again may take around 30 min to run on challenge dataset of 500 patients. 

The run logs are generated in logs/info.log

### Application configuration
The dynamic parameters are maintained in config/app_config.json
####This includes: 
1. The retry configuration parameters retry_count, backoff and delay.

2. The country_code parameter is set to "CA" to ensure our Nominatim api searches return 
results for Canada only.

3. Proximity_radius (in meters) and geohash_precision parameters required for proximity geohash search function   


## Solution approach
The overall solution to the problem of recommending nearest clinic (having shortest travel distance) to 
a patient can broadly be broken down as below:
1) Patient and Clinic address geocoding:
   We make use of the python geopy library and open source Nominatim api for geocoding patient and clinic addresses.
   There are other apis like google maps that could be used but they are paid and require the use of api keys. 
   
   ####GeoCode search approach:
   
   a) Country code is set to "CA" in app_config.json to restrict our search results to Canada
   
   b) City, Province columns as considered as fixed search criteria that is used in all searches
   
   c) Variable search criteria : For a more precise search we first use a combination of address and postal code. Various substrings
      of address are tried until search results are obtained which are then filtered on basis of postal code.
   
      If address and postal code combination search yields no result, we use a combination of postal code and fsa.
   
      If postal code and fsa combination too fails to yield any search result we just use fsa eventually falling
      back upon the usage of just fixed search criteria in step b.
   

2) Proximity clinic search using geohashes 
   
    To avoid doing brute force search on all available clinic location to get the clinic nearest to a patient 
    location, we reduce the problem to first finding clinics in proximity of the patient location using geohashing 
    technique (https://en.wikipedia.org/wiki/Geohash). We use python geolib library for geohashing a geocode.


3) Generate proximity geo hashes within a specific radius and specific precision:

    To handle the edge case of a location being on the boundary of a geohash bounding box, we need to broaden the geohash
    search space. For this we generate a list of geohashes that are with in a specific radius of the patient location.
    in addition to including the eight closest neighbours of a patient's location's geohash


4) Based on geohash matching find clinics in proximity to a patient:
   
    In step 3 we generated a list of geohashes (for a patient's location). For each of these geohashes we do a 
    string prefix search (using tries) on the clinic data set and return a set of clinics having matching geohash
    prefixes. These clinics are the clinics in proximity of the patient location.


5) Calculate travel distance from patients geolocation to each of the nearby clinics found in above step
using open source OSRM api. Again there are various options available including google maps api (which is paid)     

   
6) Pick the clinic with shortest travel distance and map it to the patient record   