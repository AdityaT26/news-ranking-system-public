# main.py
import argparse
import random
import json
import sys

from data_loader import load_data
from model_training import (
    build_feature_matrix,
    train_xgboost_model,
    save_model,
    load_model
)
from user_cohort import assign_cohorts
from article_ranking import rank_articles_for_user
from utils import remove_duplicate_users, filter_users_with_categories

# DYNAMIC COLOR CODES
COLOR_CODES = [
    "\033[91m",  # Red
    "\033[92m",  # Green
    "\033[93m",  # Yellow
    "\033[94m",  # Blue
    "\033[95m",  # Magenta
    "\033[96m",  # Cyan
    "\033[90m",  # Dark Gray
    "\033[37m",  # Light Gray
]
RESET_COLOR = "\033[0m"

N_ART = 100

SHOW_ALL_USER_PLUS_COHORTS = False

def get_dynamic_color(category_name: str) -> str:

    # Dynamically assigning color to each category name by hashing category_name
    if not category_name or category_name == "unknown":
        return RESET_COLOR
    h = abs(hash(category_name))
    idx = h % len(COLOR_CODES)
    return COLOR_CODES[idx]
# ---------------------------------------------------------------

def training_mode(use_local_json=True):
    data_dict = load_data(use_local_json=use_local_json, db=None)
    users = data_dict["users"]
    user_prefs = data_dict["user_preferences"]
    articles = data_dict["articles"]

    X, y = build_feature_matrix(users, user_prefs, articles)
    if len(X) == 0:
        print("No data for training.")
        return

    model = train_xgboost_model(X, y)
    save_model(model)
    print("\nXGBoost model training complete. Saved as trained_model.pkl")

def production_mode(use_local_json=True):
    # 1) Loading the stored trained model if found
    try:
        model = load_model()
    except:
        print("No trained model found. Please run training first.")
        return

    # 2) Loading data (local JSON files or MongoDB)
    data_dict = load_data(use_local_json=use_local_json, db=None)
    users = data_dict["users"]
    user_prefs_list = data_dict["user_preferences"]
    articles = data_dict["articles"]
    categories = data_dict["article_categories"]

    # 3) Building category map: OID -> name
    category_map = {}
    for cat in categories:
        cat_id = str(cat["_id"])
        cat_name = cat.get("name", "unknown")
        category_map[cat_id] = cat_name

    #   Also building an inverted category map
    inverted_category_map = {v: k for k, v in category_map.items()}

    # 4) Cleaning the user data
    users = remove_duplicate_users(users)
    users = filter_users_with_categories(users, user_prefs_list)

    # 5) Cohorts for users in the DB (not for custom users)
    user_cohort_map = assign_cohorts(users, user_prefs_list, category_map, n_clusters=15)

    if SHOW_ALL_USER_PLUS_COHORTS:
        for u, c in user_cohort_map.items():
            print(u, c)
        sys.exit(0)

    # 6) Mapping user_id -> user_pref to correctly fetch data
    prefs_map = {str(up["user_id"]): up for up in user_prefs_list}

    # 7) Picking 20 random DB users
    if len(users) <= 20:
        chosen_users = users
    else:
        chosen_users = random.sample(users, 20)

    print("\n--- List of 20 Random Users ---")
    for idx, user in enumerate(chosen_users):
        print(f"{idx}. User ID: {user['_id']}")

    choice = input("\nSelect user index to see details (0..19) or type 'custom': ")

    # 8) If the user picks a valid numeric index -> DB user
    try:
        choice_int = int(choice)
    except ValueError:
        choice_int = None

    if choice_int is not None and 0 <= choice_int < len(chosen_users):
        selected_user = chosen_users[choice_int]
        uid_str = str(selected_user["_id"])
        user_pref = prefs_map.get(uid_str, {})

        language_pref = user_pref.get("language", "english")
        category_oids = user_pref.get("article_category", [])
        category_names = [category_map.get(oid, "unknown") for oid in category_oids]

        cluster_top_words = user_cohort_map.get(uid_str, "general")
        assigned_cohort = f"{language_pref.lower()}-speaking {cluster_top_words} fans"

        print(f"\n--- User Details ---")
        print(f"User ID: {selected_user['_id']}")
        print(f"Language Preferences: {language_pref}")
        print(f"Interested Categories: {category_names}")
        print(f"Cohort: {assigned_cohort}")

        ranked_indices_scores = rank_articles_for_user(
            model,
            selected_user,
            prefs_map,
            articles,
            category_map=category_map,
            custom_cat_clicks=None,
            custom_tag_clicks=None
        )

        top_n = ranked_indices_scores[:N_ART]
        print("\n--- Top 100 Articles for this user ---")
        for rank, (article_idx, final_score) in enumerate(top_n, start=1):
            article = articles[article_idx]
            article_title = article.get("title", "Untitled Article")

            # Finding the article's first category name
            cat_oids = article.get("category", [])
            cat_name = "unknown"
            if cat_oids and isinstance(cat_oids, list) and "$oid" in cat_oids[0]:
                cat_oid = cat_oids[0]["$oid"]
                cat_name = category_map.get(cat_oid, "unknown")

            # Color coding the article
            color_code = get_dynamic_color(cat_name)
            print(f"{color_code}{rank}. [{final_score}/100] \"{article_title}\" ({cat_name}){RESET_COLOR}")
        return

    # 9) If the user types 'custom'
    if choice.lower() == "custom":
        try:
            with open("user_data.json", "r", encoding="utf-8") as f:
                custom_data = json.load(f)
        except FileNotFoundError:
            print("user_data.json not found.")
            return

        language_pref = custom_data.get("language", "english")
        favorite_categories = custom_data.get("favorite_categories", [])
        favorite_tags = custom_data.get("favorite_tags", [])

        # Converting [("Politics", 62), ...] -> dict { "Politics": 62, ... }
        custom_cat_clicks = {}
        for (cat_name, count) in favorite_categories:
            custom_cat_clicks[cat_name] = count

        custom_tag_clicks = {}
        for (tag_name, count) in favorite_tags:
            custom_tag_clicks[tag_name] = count

        # Creating a list of OIDs for the custom user
        category_oids = []
        for (cat_name, _count) in favorite_categories:
            oid = inverted_category_map.get(cat_name)
            if oid:
                category_oids.append(oid)

        selected_user = {
            "_id": "custom_user",
            "language": language_pref,
            "article_category": category_oids,
        }
        assigned_cohort = f"{language_pref.lower()}-speaking custom fans"

        print(f"\n--- Custom User Details ---")
        print(f"Language Preferences: {language_pref}")
        print("Favorite Categories & Clicks:")
        for cat_name, count in custom_cat_clicks.items():
            print(f"  {cat_name} => {count} clicks")
        print("Favorite Tags & Clicks:")
        for tag_name, count in custom_tag_clicks.items():
            print(f"  {tag_name} => {count} clicks")
        print(f"Cohort: {assigned_cohort}")

        # Making a mini preference map
        custom_prefs_map = {
            "custom_user": {
                "language": language_pref,
                "article_category": category_oids,
            }
        }

        # Ranking with category_map, plus custom click data
        ranked_indices_scores = rank_articles_for_user(
            model,
            selected_user,
            custom_prefs_map,
            articles,
            category_map=category_map,
            custom_cat_clicks=custom_cat_clicks,
            custom_tag_clicks=custom_tag_clicks
        )

        top_n = ranked_indices_scores[:N_ART]
        print("\n--- Top 100 Articles for this custom user ---")
        for rank, (article_idx, final_score) in enumerate(top_n, start=1):
            article = articles[article_idx]
            article_title = article.get("title", "Untitled Article")

            cat_oids = article.get("category", [])
            cat_name = "unknown"
            if cat_oids and isinstance(cat_oids, list) and "$oid" in cat_oids[0]:
                cat_oid = cat_oids[0]["$oid"]
                cat_name = category_map.get(cat_oid, "unknown")

            color_code = get_dynamic_color(cat_name)
            print(f"{color_code}{rank}. [{final_score}/100] \"{article_title}\" ({cat_name}){RESET_COLOR}")
        return

    print("Invalid choice.")

def main():
    parser = argparse.ArgumentParser(description="Personalized News Ranking System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "production"],
        required=True,
        help="Mode: training or production"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local JSON files instead of MongoDB"
    )
    args = parser.parse_args()

    if args.mode == "training":
        training_mode(use_local_json=args.local)
    elif args.mode == "production":
        production_mode(use_local_json=args.local)

if __name__ == "__main__":
    main()
