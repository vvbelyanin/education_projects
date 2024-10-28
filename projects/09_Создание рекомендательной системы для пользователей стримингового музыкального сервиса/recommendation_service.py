"""
recommendation_service.py

This module provides a recommendation service API built with FastAPI. It handles various endpoints related 
to user recommendations, including fetching popular items, similar items, user history, and processing 
event updates. The module also includes functionality for managing and retrieving recommendations based on 
user interactions and historical data.

Global Constants:
- MAX_REQUEST_RECORDS: The maximum number of records to request or return in a single API call.
- MAX_COLUMN_WIDTH: The maximum width for any text fields in the response data.
- RANDOM_SEED: Seed value for random operations to ensure reproducibility.
- S3_RECS_KEY: S3 key path for recommendations data.
- S3_DATA_KEY: S3 key path for additional data.
- LOCAL_FOLDER: Local directory for storing downloaded data.
- DATASETS: Dictionary mapping dataset names to their respective S3 and local paths.
- S3_BUCKET: Name of the S3 bucket where data is stored.
- S3_CREDS: Dictionary containing AWS S3 credentials.
- URL: The base URL for the FastAPI service.

Global Variables:
- logger: A logging instance for capturing runtime information and errors.
- recs: Instance of the Recommendations class that manages recommendation logic.

Functions:
- track_info: Processes and formats track information into a dictionary.
- get_blended_list: Merges two lists based on a probability or strict (default) alternation method.
- load: Loads a dataset from S3 or local storage.

Classes:
- EventsRequest: Pydantic model for handling event data in API requests.
- Recommendations: Core class handling the recommendation logic, user history management, and statistics tracking.

FastAPI Endpoints:
- /: Root endpoint to check the API status.
- /stats: Provides statistics about the recommendations generated.
- /top_popular: Retrieves the most popular items.
- /similar/{item_id}: Retrieves similar items for a given item ID.
- /history/{user_id}: Retrieves the history of a specified user.
- /recommendations/{user_id}: Provides recommendations for a specified user.
- /events: Accepts POST requests to update user history with new events.

Lifespan Management:
- lifespan: Async context manager for handling the startup and shutdown processes of the FastAPI application.

Main Function:
- main: Runs the Uvicorn server to host the FastAPI application.

Usage:
- To start the FastAPI server, execute this module directly. The server will be available at the specified host and port.

Example:
    $ python3 recommendation_service.py
"""
import logging
import os
import random
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Union

import boto3
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Global constants
load_dotenv()
MAX_REQUEST_RECORDS = int(os.getenv('MAX_REQUEST_RECORDS'))
MAX_COLUMN_WIDTH = 50
RANDOM_SEED = 42
S3_RECS_KEY = "recsys/recommendations"
S3_DATA_KEY = "recsys/data"
LOCAL_FOLDER = "parquet"
URL = os.getenv('BASE_URL')

DATASETS = {
    "recommendations": (
        f"{S3_RECS_KEY}/recommendations.parquet",
        f"{LOCAL_FOLDER}/recommendations.parquet"
    ),
    "similar": (
        f"{S3_RECS_KEY}/similar.parquet",
        f"{LOCAL_FOLDER}/similar.parquet"
    ),
    "top_popular": (
        f"{S3_RECS_KEY}/top_popular.parquet",
        f"{LOCAL_FOLDER}/top_popular.parquet"
    ),
    "items": (
        f"{S3_DATA_KEY}/items.parquet",
        f"{LOCAL_FOLDER}/items.parquet"
    ),
    "catalog_names": (
        f"{S3_DATA_KEY}/catalog_names.parquet",
        f"{LOCAL_FOLDER}/catalog_names.parquet"
    ),
}

S3_BUCKET = os.getenv('S3_BUCKET_NAME')
S3_CREDS = {
    "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
    "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
    "endpoint_url": os.getenv('S3_ENDPOINT_URL')
}

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# Logger setup
logger = logging.getLogger("uvicorn.error")


def track_info(
    items: pd.DataFrame,
    catalog_names: pd.DataFrame,
    idx: List[int]
) -> Dict[int, str]:
    """
    Retrieves track information based on provided item indices and returns it as a dictionary.

    :param items: DataFrame containing item data.
    :param catalog_names: DataFrame containing catalog names data.
    :param idx: List of item indices to retrieve information for.
    :return: Dictionary with track information, where each key is an order number and the value is a dictionary with track details.
    """
    df = (
        items[items['item_id'].isin(idx)]
        .merge(
            catalog_names[catalog_names['type'] == 'track'], left_on='item_id', right_on='id'
        ).drop(['id', 'type'], axis=1).rename(columns={'name': 'track'})
        .merge(
            catalog_names[catalog_names['type'] == 'album'], left_on='albums', right_on='id'
        ).drop(['albums', 'id', 'type'], axis=1).rename(columns={'name': 'album'})
        .merge(
            catalog_names[catalog_names['type'] == 'artist'], left_on='artists', right_on='id'
        ).drop(['artists', 'id', 'type'], axis=1).rename(columns={'name': 'artist'})
        .merge(
            catalog_names[catalog_names['type'] == 'genre'], left_on='genres', right_on='id'
        ).drop(['genres', 'id', 'type'], axis=1).rename(columns={'name': 'genre'})
        .set_index('item_id').loc[idx].reset_index()
        .groupby(['item_id', 'track'], sort=False).agg({
            'artist': lambda x: ", ".join(list(x.unique())),
            'album': lambda x: ", ".join(list(x.unique())),
            'genre': lambda x: ", ".join(list(x.unique()))
        }).reset_index().drop(['item_id'], axis=1)
    )

    # Convert the DataFrame to a dictionary and truncate long strings
    return {
        k: " / ".join(
            s[:MAX_COLUMN_WIDTH] + "..." if len(s) > MAX_COLUMN_WIDTH else s
            for s in v.values()
        )
        for k, v in df.to_dict(orient='index').items()
    }


def load(file_name: str) -> pd.DataFrame:
    """
    Loads a dataset from S3 or local storage.

    :param file_name: The name of the file to load.
    :return: The loaded DataFrame.
    """
    s3_key, local_path = DATASETS[file_name]

    if os.path.exists(local_path):
        logger.info(f"File {local_path} exists and loaded.")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3 = boto3.client('s3', **S3_CREDS)
        s3.download_file(S3_BUCKET, *DATASETS[file_name])
        logger.info(f"File {s3_key} downloaded from S3 storage and opened.")

    return pd.read_parquet(local_path)


def get_blended_list(
    list1: List[int],
    list2: List[int],
    prob_list1: float,
    k: int,
    choose_type: str = 'strict'
) -> List[int]:
    """
    Blends two lists based on a probability or strict alternation.

    :param list1: First list to blend.
    :param list2: Second list to blend.
    :param prob_list1: Probability of selecting an item from the first list.
    :param k: Maximum number of items in the blended list.
    :param choose_type: Method of blending ('strict' or 'probability').
    :return: Blended list of items.
    """
    unique_list1 = list(dict.fromkeys(list1))
    unique_list2 = list(dict.fromkeys(list2))
    blended_list = []

    if choose_type == 'probability':
        while unique_list1 or unique_list2:
            if random.random() < prob_list1 and unique_list1:
                blended_list.append(unique_list1.pop(0))
            elif unique_list2:
                blended_list.append(unique_list2.pop(0))
    elif choose_type == 'strict':
        while unique_list1 or unique_list2:
            if unique_list1:
                blended_list.append(unique_list1.pop(0))
            if unique_list2:
                blended_list.append(unique_list2.pop(0))

    return blended_list[:k]


class EventsRequest(BaseModel):
    """
    Request model for event data.
    """
    user_id: int
    items: List[int]


class Recommendations:
    """
    Recommendations service class.
    """
    def __init__(self):
        self.users_history = defaultdict(list)
        self.pers_recs_count = 0
        self.def_recs_count = 0
        self.similar_count = 0
        logger.info("Class Recommendations() initialized.")

    def load(self) -> None:
        """
        Loads required datasets into memory.
        """
        try:
            logger.info("Loading datasets...")
            self.recs = load("recommendations")
            self.similar = load("similar")
            self.top_popular = load("top_popular")
            self.items = load("items")
            self.catalog_names = load("catalog_names")
            logger.info("Datasets loaded.")
        except Exception as e:
            logger.error(f"Failed to load dataset(s): {e}")
            sys.exit(1)

    def get(self, user_id: int, k: int = MAX_REQUEST_RECORDS) -> Dict[int, str]:
        """
        Retrieves recommendations for a user.

        :param user_id: ID of the user.
        :param k: Number of recommendations to retrieve.
        :return: Dictionary with track information.
        """
        offline_list = list(self.recs.loc[self.recs.user_id == user_id].item_id)
        online_list = self.get_similar_items(self.users_history[user_id], k)
        blended_list = get_blended_list(offline_list, online_list, 0.5, k)

        logger.info(
            f"Getting {k} recommendations: {user_id=}: personal recs: {len(offline_list)}, "
            f"online history: {len(online_list)}."
        )

        if blended_list:
            self.pers_recs_count += 1
        else:
            self.def_recs_count += 1
            blended_list = list(self.top_popular.item_id)[:k]

        return track_info(self.items, self.catalog_names, blended_list)

    def stats(self) -> Dict[str, Union[int, List[int]]]:
        """
        Returns statistics of recommendations.

        :return: Dictionary containing statistics.
        """
        logger.info("Stats for recommendations:")
        logger.info(f"Personal recommendations: {self.pers_recs_count}")
        logger.info(f"Default recommendations: {self.def_recs_count}")
        logger.info(f"Similar tracks: {self.similar_count}")

        logger.info(f"Users with offline recommendations: {len(self.recs.user_id.unique())}")
        logger.info(f"Users with online recommendations: {len(self.users_history)}")

        return {
            'personal': self.pers_recs_count,
            'default': self.def_recs_count,
            'similar': self.similar_count,
            'random_users': random.sample(list(self.recs.user_id.unique()), 10),
            'random_items': random.sample(list(self.items.item_id.unique()), 10),
            'user_max_number': max(self.recs.user_id.max(), 0 if not self.users_history.keys() else max(self.users_history.keys()))
        }

    def get_similar_items(self, item_list: List[int], k: int = MAX_REQUEST_RECORDS) -> List[int]:
        """
        Retrieves similar items for a given list of items.

        :param item_list: List of item IDs to find similarities for.
        :param k: Number of similar items to retrieve.
        :return: List of similar item IDs.
        """
        similar_idx = list(self.similar.loc[self.similar.item1_id.isin(item_list)].item2_id)
        similar_list = list(
            self.similar.loc[self.similar.item1_id.isin(similar_idx)]
            .sort_values(by='score', ascending=False)
            .query('item2_id not in @item_list').item2_id
            .unique()[:k]
        )
        self.similar_count += 1
        return similar_list

    def update_user_events(self, user_id: int, new_items: List[int]) -> None:
        """
        Updates user history with new items.

        :param user_id: ID of the user.
        :param new_items: List of new items to add to user history.
        """
        if new_items:
            self.users_history[user_id].extend(new_items)
        logger.info(f"Added {len(new_items)} events for {user_id=}.")

    def get_user_history(self, user_id: int, last_k: Optional[int] = None) -> List[int]:
        """
        Retrieves user history.

        :param user_id: ID of the user.
        :param last_k: Number of last records to retrieve.
        :return: List of item IDs from user history.
        """
        if last_k is None:
            return self.users_history[user_id]
        return self.users_history[user_id][-last_k:]


# Initialize Recommendations service
recs = Recommendations()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application lifespan.
    """
    logger.info("Starting...")
    recs.load()
    yield
    logger.info("Stopped.")
    recs.stats()


# FastAPI application setup
app = FastAPI(title="recommendations", lifespan=lifespan)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """
    Returns the favicon.
    """
    file_path = os.path.join("static", "favicon.png")
    return FileResponse(file_path, media_type="image/png")


@app.get("/")
async def read_root() -> Dict[str, str]:
    """
    Root endpoint to check the status of the API.
    """
    return {"status": "Alive"}


@app.get("/stats")
async def get_stats() -> Dict[str, Union[int, List[int]]]:
    """
    Endpoint to retrieve statistics about recommendations.
    """
    return recs.stats()


@app.get("/top_popular")
@app.get("/top_popular/{top_k}")
async def get_top_popular(top_k: Optional[int] = MAX_REQUEST_RECORDS) -> Dict[int, str]:
    """
    Retrieves the top popular items.

    :param top_k: Number of top items to retrieve.
    :return: Dictionary with track information.
    """
    logger.info(f"A list of the top-{top_k} has been requested.")
    return track_info(recs.items, recs.catalog_names, recs.top_popular.item_id[:top_k])


@app.get("/similar/{item_id}")
@app.get("/similar/{item_id}/{top_k}")
async def get_similar(item_id: int, top_k: Optional[int] = MAX_REQUEST_RECORDS) -> Dict[int, str]:
    """
    Retrieves similar items for a given item ID.

    :param item_id: ID of the item to find similarities for.
    :param top_k: Number of similar items to retrieve.
    :return: Dictionary with track information.
    """
    logger.info(f"Similar {top_k} items have been requested for {item_id=}.")
    return track_info(recs.items, recs.catalog_names, recs.get_similar_items([item_id], top_k))


@app.get("/history/{user_id}")
@app.get("/history/{user_id}/{top_k}")
async def get_history(user_id: int, top_k: Optional[int] = MAX_REQUEST_RECORDS) -> List[int]:
    """
    Retrieves the user's history.

    :param user_id: ID of the user.
    :param top_k: Number of last records to retrieve.
    :return: List of item IDs from user history.
    """
    logger.info(f"The last {top_k} records of the user's history {user_id} have been requested.")
    return recs.get_user_history(user_id, top_k)


@app.get("/recommendations/{user_id}")
@app.get("/recommendations/{user_id}/{top_k}")
async def get_recommendations(user_id: int, top_k: Optional[int] = MAX_REQUEST_RECORDS) -> Dict[int, str]:
    """
    Retrieves recommendations for a user.

    :param user_id: ID of the user.
    :param top_k: Number of recommendations to retrieve.
    :return: Dictionary with track information.
    """
    logger.info(f"{top_k} recommendations have been requested for {user_id=}.")
    return recs.get(user_id, top_k)


@app.post("/events")
async def post_events(request: EventsRequest) -> Dict[str, str]:
    """
    Updates user history with new events.

    :param request: Request object containing user ID and item list.
    :return: Status message indicating the number of events added.
    """
    user_id = request.user_id
    items = request.items
    logger.info(f"Update of {user_id=} history for {len(items)} events has been requested.")
    recs.update_user_events(user_id, items)
    logger.info(f"{len(items)} events for {user_id=} added.")
    return {'status': f"{len(items)} events for {user_id=} added."}


def main():
    """
    Main function to run the Uvicorn server.
    """
    try:
        uvicorn.run(
            app,
            host=os.getenv('UVICORN_HOST'),
            port=int(os.getenv('UVICORN_PORT')),
            log_level="info"
        )
    except KeyboardInterrupt:
        print("Process terminated by user (Ctrl+C).")


if __name__ == "__main__":
    main()
