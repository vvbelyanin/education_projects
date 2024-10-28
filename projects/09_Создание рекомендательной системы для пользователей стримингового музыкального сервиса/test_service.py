"""
test_service.py

This module contains a set of unit tests for a FastAPI service, focusing on various API endpoints.
The tests ensure that the API behaves as expected, covering endpoints for popular items, similar items,
user history, events, and recommendations.

Every user, including new ones, may or may not have a history:
- Offline history refers to the interaction history that has been processed by the ALS recommendation 
  generation algorithm and saved in a file called 'recommendations.parquet'.
- Online history refers to new user-item interactions that occur during the service's operation. 
  These interactions exist as a list of items as long as the service is running.

Thus, there are four categories of users:
  - Users without any history;
  - Users with only offline history (which is stored in the dataset);
  - Users with only online history (which is stored as long as the service is running);
  - Users with both types of history (online history is added to the history stored in the dataset).

Global Constants:
- MAX_REQUEST_RECORDS: The maximum number of records to request in tests.
- LOG_FILE: The name of the log file where test results and details are stored.
- LOG_INDENT: The number of spaces to indent log output for readability.
- URL: The base URL for the FastAPI service being tested.
- RANDOM_SEED: The seed for random operations to ensure reproducibility.

Classes:
- TestFastAPIService: A unittest.TestCase subclass containing methods to test various API endpoints.

Logging:
- Logs test results and details to a specified log file.

Usage:
- Run this script to execute the tests and log the results to the specified log file.

Example:
    $ python3 test_service.py
"""

import unittest
import random
import requests
import json
import logging
import os
from typing import Dict, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global constants
MAX_REQUEST_RECORDS = int(os.getenv('MAX_REQUEST_RECORDS'))
URL = os.getenv('BASE_URL')

# Local constants
LOG_FILE = 'test_service.log'
LOG_INDENT = 10
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger()

def get_stats() -> Dict:
    """
    Fetches the current statistics from the FastAPI service.

    :return: A dictionary containing the statistics.
    """
    response = requests.get(f"{URL}/stats")
    return response.json()

class TestFastAPIService(unittest.TestCase):
    """
    Test suite for FastAPI service endpoints.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the class with data needed for tests.

        :return: None
        """
        cls.base_url = URL
        stats = get_stats()
        cls.random_items = stats['random_items']
        cls.random_item = stats['random_items'][0]
        cls.user_for_tests_05_06_07_09 = stats['user_max_number'] + 1234
        cls.user_for_test_08 = stats['user_max_number'] + 5678
        cls.user_for_test_10 = stats['random_users'][0]
        cls.user_for_test_11 = stats['random_users'][1]

    def check_endpoint(self, endpoint_suffix: str, expected_type: type, expected_length: int) -> Tuple[int, type, int]:
        """
        Helper method to check the response from an endpoint.

        :param endpoint_suffix: The endpoint path to test.
        :param expected_type: The expected data type of the response.
        :param expected_length: The expected length of the response data.
        :return: A tuple containing the status code, expected type, and expected length.
        """
        response = requests.get(f"{self.base_url}/{endpoint_suffix}")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, json.loads(json.dumps(data)))  # Ensure JSON is valid
        self.assertIsInstance(data, expected_type)
        self.assertEqual(len(data), expected_length)
        return response.status_code, expected_type, expected_length

    def format_output(self, dict_to_str: dict, header: str = "") -> str:
        """
        Helper method to format dictionary output for logging.

        :param dict_to_str: The dictionary to format.
        :param header: Optional header to include in the formatted string.
        :return: A formatted string ready for logging.
        """
        result = header + "\n"
        for k, v in dict_to_str.items():
            result += f"{' ' * LOG_INDENT}{k}: {v}\n"
        return result

    def test_01_root_endpoint(self) -> None:
        """
        Test the root endpoint to ensure the service is alive.
        """
        logger.info("01: Testing the root endpoint:")
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "Alive"})
        logger.info(f"Got status: {response.status_code}, response: {response.json()}")

    def test_02_top_popular_endpoint(self) -> None:
        """
        Test the top_popular endpoint for expected output.
        """
        logger.info("02: Testing the top_popular endpoint:")
        logger.info(f"Got {self.check_endpoint('top_popular', dict, MAX_REQUEST_RECORDS)}.")

    def test_03_top_popular_k_endpoint(self) -> None:
        """
        Test the top_popular endpoint with a specific number of items.
        """
        k = 10
        logger.info(f"03: Testing the {k} top_popular endpoint:")
        logger.info(f"Got {self.check_endpoint(f'top_popular/{k}', dict, k)}.")

    def test_04_similar_k_endpoint(self) -> None:
        """
        Test the similar endpoint with a specific item and number of similar items.
        """
        k = 10
        item = self.random_item
        logger.info(f"04: Testing the {k} similar endpoint with an {item=}:")
        logger.info(f"Got {self.check_endpoint(f'similar/{item}/{k}', dict, k)}.")

    def test_05_online_history_invalid_user_endpoint(self) -> None:
        """
        Test the history endpoint with a non-existent user.
        """
        user_id = self.user_for_tests_05_06_07_09

        logger.info(f"05: Testing the history endpoint with a non-existing {user_id=}:")
        logger.info(f"Got {self.check_endpoint(f'history/{user_id}', list, 0)}.")

    def test_06_events_post_endpoint(self) -> None:
        """
        Test posting events to create a new user's history.
        """
        user_id = self.user_for_tests_05_06_07_09
        random_items = self.random_items

        logger.info(f"06: Testing the post request with a new {user_id=}:")
        response = requests.post(f"{self.base_url}/events", json={"user_id": user_id, "items": random_items})
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, {'status': f"{len(random_items)} events for {user_id=} added."})
        logger.info(f"Got status: {response.status_code}, response: {data}.")

    def test_07_online_history_endpoint(self) -> None:
        """
        Test retrieving history for a user with newly added events.
        """
        k = 5
        user_id = self.user_for_tests_05_06_07_09

        logger.info(f"07: Testing the {k} history endpoint with the same new {user_id=}:")
        logger.info(f"Got {self.check_endpoint(f'history/{user_id}/{k}', list, k)}.")

    def test_08_recommendations_no_offline_no_online(self) -> None:
        """
        Test recommendations for a user with no offline personal history and no online history.
        """
        k = 5
        user_id = self.user_for_test_08
        logger.info(f"08: Testing the output of {k} recommendations for a non-existing {user_id=}:")
        logger.info(f"Got {self.check_endpoint(f'recommendations/{user_id}/{k}', dict, k)}.")

    def test_09_recommendations_no_offline_with_online(self) -> None:
        """
        Test recommendations for a user with only online history.
        """
        k = 5
        user_id = self.user_for_tests_05_06_07_09
        logger.info(
            f"09: Testing the output of {k} recommendations for {user_id=}, "
            "without an offline history, with an online history:"
        )
        logger.info(f"Got {self.check_endpoint(f'recommendations/{user_id}/{k}', dict, k)}.")

    def test_10_recommendations_with_offline_no_online(self) -> None:
        """
        Test recommendations for a user with only offline history.
        """
        k = 5
        user_id = self.user_for_test_10

        logger.info(
            f"10: Testing the output of {k} recommendations for {user_id=}, "
            "with an offline history, without an online history:"
        )
        logger.info(f"Got {self.check_endpoint(f'recommendations/{user_id}/{k}', dict, k)}.")

    def test_11_recommendations_with_offline_with_online(self) -> None:
        """
        Test recommendations for a user with both offline and online history.
        """
        k = 10
        user_id = self.user_for_test_11
        random_items = self.random_items

        recs_endpoint = f'recommendations/{user_id}/{k}'

        logger.info(
            f"11: Testing the output of {k} recommendations for {user_id=}, "
            "with an offline history and with an online history:"
        )
        logger.info(f"Got {self.check_endpoint(recs_endpoint, dict, k)}.")

        response = requests.get(f"{self.base_url}/{recs_endpoint}")
        logger.info(self.format_output(response.json(), f"Initial recommendations for {user_id=}:"))

        # Add an online history with random items
        logger.info(f"Making a post request to add {len(random_items)} events to the history of {user_id=}:")
        response = requests.post(f"{self.base_url}/events", json={"user_id": user_id, "items": random_items})
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, {'status': f"{len(random_items)} events for {user_id=} added."})
        logger.info(f"Got status: {response.status_code}, response: {data}.")

        # He becomes a user with both offline and online histories
        # Get blended recommendations
        response = requests.get(f"{self.base_url}/recommendations/{user_id}/10")
        logger.info(
            f"Repeating the output of {k} recommendations for {user_id=}, "
            "with an offline history and with an updated online history:"
        )
        logger.info(f"Got {self.check_endpoint(recs_endpoint, dict, k)}.")

        response = requests.get(f"{self.base_url}/recommendations/{user_id}/{k}")
        logger.info(self.format_output(response.json(), f"Updated recommendations for {user_id=}:"))

    def test_12_stats_endpoint(self) -> None:
        """
        Test the stats endpoint for expected output.
        """
        logger.info("12: Testing the stats endpoint:")
        response = requests.get(f"{self.base_url}/stats")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 6)

        data = {key: data[key] for key in ['personal', 'default', 'similar']}
        logger.info(f"Got status: {response.status_code}, response: {data}")

def main() -> None:
    """
    Main function to run the tests and log output to a file.
    """
    with open(LOG_FILE, 'a') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)

    with open(LOG_FILE, 'r') as file:
        print(f"Content of {LOG_FILE}:")
        print(file.read())

if __name__ == "__main__":
    main()
