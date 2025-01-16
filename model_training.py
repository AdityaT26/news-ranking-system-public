import pickle
import numpy as np
import random
from tqdm import tqdm
from xgboost import XGBRegressor
import datetime

from utils import (
    remove_duplicate_users,
    filter_users_with_categories,
    build_user_article_feature
)

def build_feature_matrix(users, user_prefs, articles):
    """
    Builds a more complex partial-label dataset.
    1) Identify the LATEST article's updated_at date for 'freshness' reference.
    2) Build a global category frequency map for weighting partial labels by popularity.
    3) Incorporate a bigger random range to introduce more variance.
    """
    # Finding the latest updatedAt among all articles to get the date of the latest article
    max_dt = None
    for art in articles:
        updated_at_val = art.get("updatedAt")
        article_dt = _parse_date(updated_at_val)
        if (max_dt is None) or (article_dt > max_dt):
            max_dt = article_dt

    # Fallback to a default date if no dates found
    if max_dt is None:
        max_dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

    # Building global category frequency map across all articles
    cat_frequency = {}
    total_articles = 0

    for art in articles:
        cat_list = art.get("category", [])
        used_cats = []
        for c in cat_list:
            oid = c.get("$oid")
            if oid:
                used_cats.append(oid)
        unique_cats = set(used_cats)
        for cid in unique_cats:
            cat_frequency[cid] = cat_frequency.get(cid, 0) + 1
        total_articles += 1

    # Converting frequency to [0..1] by dividing by the max
    if cat_frequency:
        max_freq = max(cat_frequency.values())
    else:
        max_freq = 1

    for cid in cat_frequency:
        cat_frequency[cid] = cat_frequency[cid] / max_freq

    # Creating matrix X, y with partial labeling
    users = remove_duplicate_users(users)
    users = filter_users_with_categories(users, user_prefs)
    prefs_map = {str(up["user_id"]): up for up in user_prefs}

    data = []
    labels = []

    total_iterations = len(users) * len(articles)
    pbar = tqdm(total=total_iterations, desc="Building feature matrix", ncols=80)

    for user in users:
        for article in articles:

            # features = [language_feature, lang_match, cat_overlap, days_old]
            features = build_user_article_feature(user, prefs_map, article)
            data.append(features)

            updated_at_val = article.get("updatedAt")
            article_dt = _parse_date(updated_at_val)
            days_diff = (max_dt - article_dt).days  # smaller -> more fresh

            # we can override the features array or handle it in the partial labeling logic directly

            # Figuring out the article's primary category -> frequency factor
            cat_list = article.get("category", [])
            freq_factor = 0.0
            if cat_list:

                # If multiple categories are present, we use all of them
                sum_factors = 0.0
                for cdict in cat_list:
                    cid = cdict.get("$oid")
                    if cid:
                        sum_factors += cat_frequency.get(cid, 0.0)
                freq_factor = sum_factors / len(cat_list)
            # freq_factor in [0..1], higher -> more popular categories

            # Partial label logic
            lang_match = features[1]  # 0 or 1
            cat_overlap = features[2] # int

            # Freshness factor: smaller days_diff -> bigger impact on freshness
            base_freshness = 0.9 - (days_diff / 180.0) * (0.9 - 0.3)
            if base_freshness < 0.3:
                base_freshness = 0.3
            if base_freshness > 0.9:
                base_freshness = 0.9

            # Base engagement
            if lang_match == 1 and cat_overlap > 0:

                base_engagement = random.uniform(0.6, 1.0)
                # scaling by cat_overlap
                overlap_scale = min(cat_overlap, 5) / 5.0
                base_engagement *= (0.5 + 0.5 * overlap_scale)
                base_engagement = min(base_engagement, 1.0)
            else:
                base_engagement = random.uniform(0.0, 0.4)

            raw_label = base_engagement * base_freshness * freq_factor

            labels.append(raw_label)
            pbar.update(1)

    pbar.close()

    X = np.array(data, dtype=float)
    y = np.array(labels, dtype=float)
    return X, y

def train_xgboost_model(X, y):

    print("Training XGBoost regressor...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse"
    )
    model.fit(X, y)
    return model

def save_model(model, path="trained_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="trained_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def _parse_date(updated_at_val):

    import datetime
    if isinstance(updated_at_val, dict):
        updated_str = updated_at_val.get("$date", "")
    elif isinstance(updated_at_val, str):
        updated_str = updated_at_val
    else:
        updated_str = ""

    try:
        dt = datetime.datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
    except:
        dt = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    return dt
