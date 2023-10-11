# $pip install geopy
from geopy.geocoders import Nominatim

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
    distances = []
    for idx, row in dataset.iterrows():
        point_lat = row['latitude']
        point_lon = row['longitude']
        dist = haversine(lat, lon, point_lat, point_lon)
        distances.append((idx, dist))
    
    # Sort by distance and get the indexes of the closest houses
    distances.sort(key=lambda x: x[1])
    closest_indexes = [idx for idx, _ in distances[:1]][0]


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
    '''
    distance in km to the nearest coal mine
    '''
    df_coal = pd.read_excel('additional_data/geographical_context.xls', sheet_name='coal_mines')
    closest_index = find_closest_line(df_coal, lat, lon)
    lat_coal = df_coal.loc[i]['latitude']
    lon_coal = df_coal.loc[i]['longitude']
    distance = haversine(lat, lon, lat_coal, lon_coal)
    return distance


def get_refinerie_factor(lat, lon):
    '''
    distance in km to the nearest refinery
    '''
    df_refineries = pd.read_excel('additional_data/geographical_context.xls', sheet_name='refineries')
    closest_index = find_closest_line(df_refineries, lat, lon)
    lat_raf = df_refineries.loc[i]['latitude']
    lon_raf = df_refineries.loc[i]['longitude']
    distance = haversine(lat, lon, lat_raf, lon_raf)
    return distance


def get_waste_factor(lat, lon):
    '''
    waste factor of the country
    '''
    df_waste = pd.read_excel('additional_data/geographical_context.xls', sheet_name='waste_management')
    country = get_country_from_coordinates(lat, lon)
    waste_score = df_waste[df_waste['country']==country]
    if len(waste_score)==0:
        waste_score = 0
    else : 
        waste_score = float(waste_score['waste_management_score'])
    return waste_score