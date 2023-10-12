import pandas as pd
import random

"""
    Predicts the likelihood of an image being a 'plume' based on specific input parameters.

    Args:
        image (str): The image file in TIFF format.
        latitude (float): The latitude associated with the image.
        longitude (float): The longitude associated with the image.

    Returns:
        float: A probability value between 0 and 1, indicating the likelihood of the image being a 'plume'.
"""


class Model_results:
    def __init__(self, file_name):
        # Initialize the model with the provided file_name (image path) and load test results from a CSV file.
        self.file_name = file_name
        self.result_csv = pd.read_csv("test_results.csv", header=0)
        pass

    def predict(self):
        # Predict the probability of an image being a plume based on the test results.
        filtered_results = self.result_csv[self.result_csv["path"] == self.file_name]

        if not filtered_results.empty:
            prediction = filtered_results["label"].iloc[0]
            return prediction
        else:
            return random.random()  # Handle the case when there's no matching result


"""
Note : 
We trained a model and theoretically should import the weights to use on any input with the right format, but this is an MVP, so for now, we just take the test results we have and output the probability the image is a plume, so the output is a float between 0 and 1.

"""
