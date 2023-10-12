import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from meteostat import Stations, Daily
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image
import imageio
import cv2
import io
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression
import tifffile as tiff
import matplotlib.pyplot as plt
import datetime as dt

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

def image_entropy(image):
    # Convertir l'image en niveaux de gris si ce n'est pas déjà le cas
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    # Calculer l'histogramme des niveaux de gris
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))

    # Normaliser l'histogramme pour obtenir une distribution de probabilité
    histogram = histogram / float(np.sum(histogram))

    # Calculer l'entropie en utilisant la formule de Shannon
    entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))

    return entropy


def compute_entropy_list(train_data_dir, metadata):
    '''returns a list of entropy calculated for each image of images_dir,
        which file is in metadata.path column'''
    entropy_list = []

    for filename in metadata.path:
        img_path = train_data_dir+'/'+filename
        img = Image.open(img_path)
        img_array = np.array(img)

    
        entropy_value = image_entropy(img_array)
        entropy_list.append(entropy_value)

    return(entropy_list)

def image_skewness(image):
    # Calculer la skewness des niveaux de gris
    skewness = skew(image, axis=None)

    return skewness

def compute_skewness_list(train_data_dir, metadata):
    '''Get skewness for each picture'''
    skewness_list = []

    for filename in metadata.path:
        img_path = train_data_dir+'/'+filename
        img = Image.open(img_path)
        img_array = np.array(img)

    
        skewness_value = image_skewness(img_array)
        skewness_list.append(skewness_value)

    return(skewness_list)

def image_kurtosis(image):
    # Convertir l'image en niveaux de gris si ce n'est pas déjà le cas
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    # Calculer le kurtosis des niveaux de gris
    kurtosis_value = kurtosis(image, axis=None)

    return kurtosis_value

def compute_kurtosis_list(train_data_dir, metadata):
    '''Get kurtosis for each image as a list'''
    kurtosis_list = []

    for filename in metadata.path:
        img_path = train_data_dir+'/'+filename
        img = Image.open(img_path)
        img_array = np.array(img)

    
        kurtosis_value = image_kurtosis(img_array)
        kurtosis_list.append(kurtosis_value)

    return(kurtosis_list)

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

    data = Daily(station, date, date)
    data = data.fetch()
    w_speed = data['wspd'].values[0]
    w_dir = data['wdir'].values[0]
    
    return w_speed, w_dir

def compute_plume_wind_direction(train_data_dir, metadata):
    
    plume_direction_list = []
    for i in range(len(metadata)):
        filename = metadata.loc[i]['path']
        img_path = train_data_dir+'/'+filename
        img = Image.open(img_path)
        image = np.array(img)
    
        max_value = np.max(image)

        ## create the mask regarding the max intensity of the image, the threshold
        seuil = 0.75*max_value
        resultat = np.where(image < seuil, 0, 1)
        resultat = resultat.astype(np.uint8)
        
        # Assurez-vous que l'image est au format uint8 (entier non signé 8 bits)
        mask = resultat

        # Trouver les contours des objets dans l'image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initialiser des variables pour suivre l'objet avec la plus grande aire
        max_area = 0
        max_contour = None

        # Parcourir tous les contours et trouver l'objet avec la plus grande aire
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            # Trouver les coordonnées (x, y) des pixels de l'objet
            coordinates = np.argwhere(mask == 1)
            x = coordinates[:, 1].reshape(-1, 1)
            y = coordinates[:, 0]

            regression = LinearRegression().fit(x, y)
            slope = regression.coef_[0]

            # Calculer l'angle en radians à partir du coefficient directeur
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)

            # Ajuster l'angle pour qu'il soit par rapport à l'axe nord-sud dans le sens horaire
            angle_deg_from_north = 90 + angle_deg

            # Si l'angle est négatif, ajouter 360 degrés pour obtenir une valeur positive dans le sens horaire
            if angle_deg_from_north < 0:
                angle_deg_from_north += 360
    
            lat = metadata.loc[i]['lat']
            lon = metadata.loc[i]['lon']
            date = metadata.loc[i]['date']
            wind_dir = get_wind_infos(lat, lon, date)[1]
            
            angle_plume_wind = np.abs(angle_deg_from_north - wind_dir)
            
            # on recherche à quel point les 2 directions sont alignées, 
            # donc on vérifie si la différence > 90°
            if angle_plume_wind > 90 : angle_plume_wind -= 90
            
            plume_direction_list.append(angle_plume_wind)
            
        else:
            
            plume_direction_list.append(None)
            
    return(plume_direction_list)

def compute_plume_parameters(train_data_dir, metadata):
    '''Get list of plume parameters (width, height, area, centroid)'''
    width_list = []
    height_list = []
    area_list = []
    centroid_x_list = []
    centroid_y_list = []

    for filename in metadata.path:
        img_path = train_data_dir+'/'+filename
        img = Image.open(img_path)
        image = np.array(img)

        max_value = np.max(image)

        ## create the mask regarding the max intensity of the image, the threshold
        seuil = 0.8*max_value
        resultat = np.where(image < seuil, 0, 1)
        resultat = resultat.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(resultat, connectivity=4)

        plus_grande_aire = 0
        label_plus_grande_aire = None

        for label in range(1, num_labels):  
            left, top, width, height, area = stats[label]
            
            if area > plus_grande_aire:
                plus_grande_aire = area
                label_plus_grande_aire = label

        # Vérifie si au moins une composante connexe a été trouvée
        if label_plus_grande_aire is not None:
            left, top, width, height, area = stats[label_plus_grande_aire]
            centroid_x, centroid_y = centroids[label_plus_grande_aire]
            width_list.append(width)
            height_list.append(height)
            area_list.append(area)
            centroid_x_list.append(centroid_x)
            centroid_y_list.append(centroid_y)
        
        else:
            print("Aucune composante connexe n'a été trouvée.")

    return(width_list, height_list, area_list, centroid_x_list, centroid_y_list)

# Read metadata for both training and test data
train_metadata = pd.read_csv('../cleanr/train data/metadata.csv')
test_metadata = pd.read_csv('../cleanr/test data/metadata.csv')

#Add tif at the end of path
train_metadata['path'] = train_metadata['path'].apply(lambda x: x + '.tif')

# Path to the directory containing image files
image_directory = '../cleanr/test data/images'

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

# Add other functions as list types
def compute_metadata_and_add_to_dataframe(data_directory, metadata_dataframe, metadata_column_name, compute_function):
    file_list = os.listdir(data_directory)
    metadata_list = compute_function(data_directory, metadata_dataframe)
    metadata_dataframe[metadata_column_name] = metadata_list

compute_metadata_and_add_to_dataframe('../cleanr/train data', train_metadata, 'entropy', compute_entropy_list)
compute_metadata_and_add_to_dataframe('../cleanr/test data/images', test_metadata, 'entropy', compute_entropy_list)
compute_metadata_and_add_to_dataframe('../cleanr/train data', train_metadata, 'skewness', compute_skewness_list)
compute_metadata_and_add_to_dataframe('../cleanr/test data/images', test_metadata, 'skewness', compute_skewness_list)
compute_metadata_and_add_to_dataframe('../cleanr/train data', train_metadata, 'kurtosis', compute_kurtosis_list)
compute_metadata_and_add_to_dataframe('../cleanr/test data/images', test_metadata, 'kurtosis', compute_kurtosis_list)
compute_metadata_and_add_to_dataframe('../cleanr/train data', train_metadata, 'plume_wind_direction', compute_plume_wind_direction)
compute_metadata_and_add_to_dataframe('../cleanr/test data/images', test_metadata, 'plume_wind_direction', compute_plume_wind_direction)

# Add plume parameters features
plume_parameters = compute_plume_parameters('../cleanr/train data', train_metadata)

# Assign the calculated plume parameters to the train_metadata dataframe
train_metadata = train_metadata.assign(
    width=plume_parameters[0],
    height=plume_parameters[1],
    area=plume_parameters[2],
    centroid_x=plume_parameters[3],
    centroid_y=plume_parameters[4]
)

# For the test_metadata dataframe, you can do the same
plume_parameters = compute_plume_parameters('../cleanr/test data/images', test_metadata)
test_metadata = test_metadata.assign(
    width=plume_parameters[0],
    height=plume_parameters[1],
    area=plume_parameters[2],
    centroid_x=plume_parameters[3],
    centroid_y=plume_parameters[4]
)

# Drop duplicates in test and train
train_metadata = train_metadata.drop_duplicates()
test_metadata = test_metadata.drop_duplicates()

#Add probability of cnn model as feature
test_results = pd.read_csv('test_results.csv')
test_metadata = test_metadata.merge(test_results, on='path', how='left')

# 2. Read the train_results CSV file, filter, and extract the path
train_results = pd.read_csv('train_results.csv')
train_results = train_results[train_results['path'].str.contains('augmented_0_')]
train_results['extracted_path'] = train_results['path'].str[-11:]

# 3. Join the extracted path with train_metadata
train_metadata['extracted_path'] = train_metadata['path'].str[-11:]
train_metadata = train_metadata.merge(train_results[['extracted_path', 'new_feature']], on='extracted_path', how='left')

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
X_train = train_metadata.drop(['plume_yes', 'set', 'path', 'id_coord', 'plume_no','date'], axis=1)
y_train = train_metadata['plume_yes']

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Predict probabilities for the test metadata
X_test = test_metadata.drop(['id_coord','path','date'], axis=1)
test_probabilities = model.predict_proba(X_test)[:, 1]

# Create a DataFrame for the output
output_df = pd.DataFrame({'path': test_metadata['path'], 'label': test_probabilities})

# Save the DataFrame to a CSV file
output_df.to_csv('test_results_ml.csv', index=False)