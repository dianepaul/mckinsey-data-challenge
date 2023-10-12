import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from meteostat import Stations, Daily
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface
    given their latitude and longitude in decimal degrees.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    
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

def find_closest_site(latitude, longitude, dataframe):
    """
    Find the closest site (row) in the dataframe to the given latitude and longitude.
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
    location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True, language="en")
    
    if location and 'address' in location.raw:
        country = location.raw['address'].get('country')
        if country:
            return country
    return "NA"

def get_coal_factor(lat, lon, df_coal):
    """
    Calculate the distance in kilometers to the nearest coal mine.
    """
    closest_index = find_closest_site(lat, lon, df_coal)
    lat_coal = df_coal.loc[closest_index]['latitude']
    lon_coal = df_coal.loc[closest_index]['longitude']
    distance = haversine(lat, lon, lat_coal, lon_coal)
    return distance

def get_refinerie_factor(lat, lon, df_refineries):
    """
    Calculate the distance in kilometers to the nearest refinery.
    """
    closest_index = find_closest_site(lat, lon, df_refineries)
    lat_raf = df_refineries.loc[closest_index]['latitude']
    lon_raf = df_refineries.loc[closest_index]['longitude']
    distance = haversine(lat, lon, lat_raf, lon_raf)
    return distance

def get_waste_factor(lat, lon, df_waste):
    """
    Get the waste factor of the country.
    """
    country = get_country_from_coordinates(lat, lon)
    waste_score = df_waste[df_waste['country'] == country]
    return float(waste_score['waste_management_score']) if not waste_score.empty else 0

def get_methane_factor(lat, lon, df_methane):
    country = get_country_from_coordinates(lat, lon)
    return (
        df_methane['emissions'].mean()
        if country not in df_methane.index.tolist()
        else df_methane.loc[country]['emissions']
    )

def get_wind_infos(lat, lon, date, stations):
    """
    Get wind speed and wind direction for the specified location and date.
    """
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
    date = dt.datetime(year, month, day)

    stations = stations.nearby(lat, lon)
    stations = stations.fetch()
    stations = stations[stations['daily_end'] >= date]
    station = stations.index[0]

    data = Daily(station, date, date)
    data = data.fetch()
    w_speed = data['wspd'].values[0]
    w_dir = data['wdir'].values[0]
    
    return w_speed, w_dir

# Read metadata for both training and test data
train_metadata = pd.read_csv('../cleanr/train data/metadata.csv')
test_metadata = pd.read_csv('../cleanr/test data/metadata.csv')
test_metadata = test_metadata.drop_duplicates()

# Path to the directory containing image files
image_directory = 'cleanr/test data/images'

# Function to find the filename based on 'id_coord'
def find_filename(id_coord):
    for filename in os.listdir(image_directory):
        if f"{id_coord}.tif" in filename:
            return filename
    return None

# Apply the function to populate the 'path' column
test_metadata['path'] = test_metadata['id_coord'].apply(find_filename)

# Load additional data
df_coal = pd.read_excel('../additional_data/geographical_context.xls', sheet_name='coal_mines')
df_refineries = pd.read_excel('../additional_data/geographical_context.xls', sheet_name='refineries')
df_waste = pd.read_excel('../additional_data/geographical_context.xls', sheet_name='waste_management')
df_methane = pd.read_csv('../additional_data/aggregated_methane.csv')
df_methane.set_index('country', inplace=True)
stations = Stations()

# Calculate features for both training and test data
for metadata in [train_metadata, test_metadata]:
    metadata['coal_factor'] = metadata.apply(lambda row: get_coal_factor(row['lat'], row['lon'], df_coal), axis=1)
    metadata['refinerie_factor'] = metadata.apply(lambda row: get_refinerie_factor(row['lat'], row['lon'], df_refineries), axis=1)
    metadata['methane_factor'] = metadata.apply(lambda row: get_methane_factor(row['lat'], row['lon'], df_methane), axis=1)
    metadata['waste_factor'] = metadata.apply(lambda row: get_waste_factor(row['lat'], row['lon'], df_waste), axis=1)

# One-hot encoding for categorical feature 'plume'
train_metadata = pd.get_dummies(train_metadata, columns=['plume'])

# Define hyperparameters for the XGBoost classifier
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.15,
    'n_estimators': 100,
    'eval_metric': 'logloss'
}

# Train the XGBoost classifier on the entire training dataset
X_train = train_metadata.drop(['plume_yes', 'set', 'path', 'id_coord', 'plume_no'], axis=1)
y_train = train_metadata['plume_yes']

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Predict probabilities for the test metadata
X_test = test_metadata.drop(['id_coord','path'], axis=1)
test_probabilities = model.predict_proba(X_test)[:, 1]

# Create a DataFrame for the output
output_df = pd.DataFrame({'path': test_metadata['path'], 'label': test_probabilities})

# Save the DataFrame to a CSV file
output_df.to_csv('test_results_ml.csv', index=False)