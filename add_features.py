# $pip install geopy
# $pip install meteostat

from geopy.geocoders import Nominatim
from meteostat import Stations, Daily
import pandas as pd
import datetime as dt

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude
    in decimal degrees.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of the Earth in kilometers (mean value)
    radius = 6371.0
    
    # Calculate the distance
    distance = radius * c
    return distance

def find_closest_lines(dataset, lat, lon, num_closest=3):
    '''
    INPUTS : 
        - dataset with columns 'latitude' and 'longitude'
        - latitude of a given point
        - longitude of a given point
        - num_closest : the number of results to return

    OUTPUT : the indexes of the closest points in the dataset
    '''

    distances = []
    for idx, row in dataset.iterrows():
        point_lat = row['latitude']
        point_lon = row['longitude']
        dist = haversine(lat, lon, point_lat, point_lon)
        distances.append((idx, dist))
    
    # Sort by distance and get the indexes of the closest houses
    distances.sort(key=lambda x: x[1])
    closest_indexes = [idx for idx, _ in distances[:num_closest]]
    
    return closest_indexes

def find_closest_line(dataset, lat, lon):
    '''
    INPUTS : 
        - dataset with columns 'latitude' and 'longitude'
        - latitude of a given point
        - longitude of a given point

    OUTPUT : the index of the closest point in the dataset
    '''
    closest_index = find_closest_lines(dataset, lat, lon, num_closest=1)[0]
    return(closest_index)

def get_country_from_coordinates(latitude, longitude):
    """
    Get the country name from latitude and longitude coordinates.
    """
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True, language="en")
    
    if location and 'address' in location.raw:
        country = location.raw['address'].get('country')
        if country:
            return country
    return "NA"


df_refineries = pd.read_excel('additional_data/geographical_context.xls', sheet_name='refineries')
df_coal = pd.read_excel('additional_data/geographical_context.xls', sheet_name='coal_mines')
df_waste = pd.read_excel('additional_data/geographical_context.xls', sheet_name='waste_management')
df_methane = pd.read_csv('additional_data/aggregated_methane.csv')
df_methane.set_index('country', inplace=True)


def get_coal_factor(lat, lon, df_coal):
    '''
    distance in km to the nearest coal mine
    '''
    closest_index = find_closest_line(df_coal, lat, lon)
    lat_coal = df_coal.loc[closest_index]['latitude']
    lon_coal = df_coal.loc[closest_index]['longitude']
    distance = haversine(lat, lon, lat_coal, lon_coal)
    return distance

def get_refinerie_factor(lat, lon, df_refineries):
    '''
    distance in km to the nearest refinery
    '''
    closest_index = find_closest_line(df_refineries, lat, lon)
    lat_raf = df_refineries.loc[closest_index]['latitude']
    lon_raf = df_refineries.loc[closest_index]['longitude']
    distance = haversine(lat, lon, lat_raf, lon_raf)
    return distance

def get_waste_factor(lat, lon, df_waste):
    '''
    waste factor of the country
    '''
    country = get_country_from_coordinates(lat, lon)
    waste_score = df_waste[df_waste['country']==country]
    return (
        waste_score['waste_management_score'].mean()
        if len(waste_score) == 0
        else float(waste_score['waste_management_score'])
    )

def get_methane_factor(lat, lon, df_methane):
    country = get_country_from_coordinates(lat, lon)
    return (
        df_methane['emissions'].mean()
        if country not in df_methane.index.tolist()
        else df_methane.loc[country]['emissions']
    )

def get_wind_infos(lat, lon, date):
    '''
    INPUT : 
        - lat : latitude
        - lon : longitude
        - date (int format : yyyymmdd)
        
    OUTPUT : 
        tuple (wind speed, wind direction) 
        for the specified location at the specified date
        (the infos correspond to the closest meterological station
        with available data for the specified date)
    '''

    date = str(date)
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
    date = dt.datetime(year, month, day)

    stations = Stations()
    stations = stations.nearby(lat, lon)
    stations = stations.fetch()
    stations = stations[stations['daily_end']>=date]
    station = stations.index[0]

    data = Daily(station, start, start)
    data = data.fetch()
    w_speed = data['wspd'].values[0]
    w_dir = data['wdir'].values[0]
    
    return w_speed, w_dir