import pandas as pd
import numpy as np
import datetime as dt
from geopy.geocoders import Nominatim
from meteostat import Stations, Daily
from scipy.optimize import minimize
import xgboost as xgb

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
    """
    Find the indexes of the closest points in the dataset to a given location.
    """
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

def find_closest_site(latitude, longitude, dataframe):
    """
    Find the closest site in the dataframe to a given latitude and longitude.
    """
    closest_distance = float('inf')  # Initialize with infinity
    closest_site_index = None

    for idx, row in dataframe.iterrows():
        lat, lon = row['latitude'], row['longitude']
        distance = haversine(latitude, longitude, lat, lon)
        if distance < closest_distance:
            closest_distance = distance
            closest_site_index = idx

    return closest_site_index

def get_country_from_coordinates(latitude, longitude):
    """
    Get the country name from latitude and longitude coordinates.
    """
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
    
    if location and 'address' in location.raw:
        country = location.raw['address'].get('country')
        if country:
            return country
    return "Country not found"

def get_coal_factor(lat, lon):
    """
    Get the distance in km to the nearest coal mine.
    """
    df_coal = pd.read_excel('additional_data/geographical_context.xls', sheet_name='coal_mines')
    closest_index = find_closest_site(lat, lon, df_coal)
    lat_coal = df_coal.loc[closest_index]['latitude']
    lon_coal = df_coal.loc[closest_index]['longitude']
    distance = haversine(lat, lon, lat_coal, lon_coal)
    return distance

def get_refinerie_factor(lat, lon):
    """
    Get the distance in km to the nearest refinery.
    """
    df_refineries = pd.read_excel('additional_data/geographical_context.xls', sheet_name='refineries')
    closest_index = find_closest_site(lat, lon, df_refineries)
    lat_raf = df_refineries.loc[closest_index]['latitude']
    lon_raf = df_refineries.loc[closest_index]['longitude']
    distance = haversine(lat, lon, lat_raf, lon_raf)
    return distance

def get_waste_factor(lat, lon):
    """
    Get the waste factor of the country.
    """
    df_waste = pd.read_excel('additional_data/geographical_context.xls', sheet_name='waste_management')
    country = get_country_from_coordinates(lat, lon)
    waste_score = df_waste[df_waste['country']==country]
    if len(waste_score)==0:
        waste_score = 0
    else : 
        waste_score = float(waste_score['waste_management_score'])
    return waste_score


def get_wind_infos(lat, lon, date):
    """
    Get wind speed and direction for a specified location and date.
    """
    date = str(date)
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
    date = dt.datetime(year, month, day)

    stations = Stations()
    stations = stations.nearby(lat, lon)
    stations = stations.fetch()
    stations = stations[stations['daily_end']>=date]
    station = stations.index[0]

    data = Daily(station, date, date)
    data = data.fetch()
    w_speed = data['wspd'].values[0]
    w_dir = data['wdir'].values[0]
    
    return w_speed, w_dir

# Load the training metadata
path_metadata = 'cleanr/train data/metadata.csv'
metadata_train = pd.read_csv(path_metadata)

# Load the test metadata
path_metadata_test = 'cleanr/test data/metadata.csv'
metadata_test = pd.read_csv(path_metadata_test)

# Combine training and test metadata
metadata = pd.concat([metadata_train, metadata_test], ignore_index=True)

# Add features to the metadata
metadata['country'] = metadata.apply(lambda row: get_country_from_coordinates(row['lat'], row['lon']), axis=1)
metadata['coal_factor'] = metadata.apply(lambda row: get_coal_factor(row['lat'], row['lon']), axis=1)
metadata['refinerie_factor'] = metadata.apply(lambda row: get_refinerie_factor(row['lat'], row['lon']), axis=1)
metadata['waste_factor'] = metadata.apply(lambda row: get_waste_factor(row['lat'], row['lon']), axis=1)
metadata['w_speed'] = metadata.apply(lambda row: get_wind_infos(row['lat'], row['lon'], row['date']), axis=1)
metadata['w_speed'] = metadata['w_speed'].apply(lambda x: float(str(x).split(',')[0][1:]))