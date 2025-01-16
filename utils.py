import datetime

def remove_duplicate_users(users):
    unique = {}
    for user in users:
        uid = user.get("user_id") or user.get("_id")
        if uid not in unique:
            unique[uid] = user
    return list(unique.values())

def filter_users_with_categories(users, user_preferences):
    valid_users = []
    prefs_map = { str(up["user_id"]): up for up in user_preferences }
    for user in users:
        user_id = str(user.get("_id")) or user.get("user_id")
        pref = prefs_map.get(user_id)
        if pref and pref.get("article_category"):
            valid_users.append(user)
    return valid_users

def build_user_article_feature(user, prefs_map, article):
    """
    Returns a 4-feature vector:
      [language_feature, lang_match, cat_overlap, days_old].
    """
    user_id_str = str(user["_id"])
    user_pref = prefs_map.get(user_id_str, {})
    user_language = user_pref.get("language", "english").lower()
    language_feature = 1 if user_language == "english" else 2

    article_lang = article.get("language", "english").lower()
    lang_match = 1 if user_language == article_lang else 0

    user_categories = set(user_pref.get("article_category", []))
    article_cat_oids = set()
    for c in article.get("category", []):
        if "$oid" in c:
            article_cat_oids.add(c["$oid"])
    cat_overlap = len(user_categories.intersection(article_cat_oids))

    updated_at_val = article.get("updatedAt")  # could be a string or a dict
    updated_at_str = ""

    if isinstance(updated_at_val, dict):
        # If it's a dict with {"$date": "..."} structure
        updated_at_str = updated_at_val.get("$date", "")
    elif isinstance(updated_at_val, str):
        # If it's already a string
        updated_at_str = updated_at_val
    else:
        # Fallback if it's None or unexpected
        updated_at_str = ""

    try:
        updated_at_dt = datetime.datetime.fromisoformat(
            updated_at_str.replace("Z", "+00:00")
        )
    except:
        updated_at_dt = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)

    # Making 'now' offset-aware to avoid subtracting naive vs. aware
    now = datetime.datetime.now(datetime.timezone.utc)
    days_old = (now - updated_at_dt).days
    if days_old < 0:
        days_old = 0

    return [language_feature, lang_match, cat_overlap, days_old]