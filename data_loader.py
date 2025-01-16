import json
import os
from typing import List, Dict
from pymongo.collection import Collection

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

def load_local_json(filename: str) -> List[Dict]:
    """
    Load JSON from data/ folder. Returns a list of dict objects.
    """
    file_path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # If the JSON is a dict, wrap it into a list
        if isinstance(data, dict):
            data = [data]
        return data

def fetch_collection_as_list(collection: Collection) -> List[Dict]:
    # Returning the entire collection as a list of disctionaries
    return list(collection.find({}))

def load_data(use_local_json=True, db=None) -> Dict[str, List[Dict]]:
    """
    Main data loading function:
      - If use_local_json=True, loads data from local JSON files in the data folder.
      - Otherwise, fetch from MongoDB database.

    Structure of the returned dictionary:
      {
         'users': [...],
         'user_preferences': [...],
         'articles': [...],
         'article_categories': [...],
         'tags': [...],
         'tag': [...]
      }
    """
    data_dict = {}
    if use_local_json:
        # Loading data from local data folder
        data_dict["users"] = load_local_json("users.json")
        data_dict["user_preferences"] = load_local_json("user_preferences.json")
        data_dict["articles"] = load_local_json("articles.json")
        data_dict["article_categories"] = load_local_json("article_categories.json")
        data_dict["tags"] = load_local_json("tags.json")  # or if you have both tags.json and tag.json
        data_dict["tag"] = load_local_json("tag.json")
    else:
        # Loading data from MongoDB
        data_dict["users"] = fetch_collection_as_list(db["users"])
        data_dict["user_preferences"] = fetch_collection_as_list(db["user_preferences"])
        data_dict["articles"] = fetch_collection_as_list(db["articles"])
        data_dict["article_categories"] = fetch_collection_as_list(db["article_categories"])
        data_dict["tags"] = fetch_collection_as_list(db["tags"])
        data_dict["tag"] = fetch_collection_as_list(db["tag"])

    return data_dict
