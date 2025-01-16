import numpy as np
from utils import build_user_article_feature

def rank_articles_for_user(
    model,
    user,
    prefs_map,
    articles,
    category_map=None,
    custom_cat_clicks=None,
    custom_tag_clicks=None,
    alpha=8.0,              # User-Click influence multiplier
    k=25.0,                 # Scale factor for diminishing returns
    novelty_boost=15.0      # Novelty boost bonus if user_clicks ~ 0
):

    # Base model predictions (scaled between 0 and 100)
    all_features = []
    for article in articles:
        feat = build_user_article_feature(user, prefs_map, article)
        all_features.append(feat)

    X = np.array(all_features, dtype=float)
    predicted_scores = model.predict(X)
    base_scaled_scores = min_max_scale(predicted_scores)

    # If not a custom user -> just return base scores
    if str(user["_id"]) != "custom_user" or not custom_cat_clicks or not custom_tag_clicks:
        return sorted(enumerate(base_scaled_scores), key=lambda x: x[1], reverse=True)

    final_scores = []
    user_lang = user.get("language", "english").lower()

    for i, article in enumerate(articles):
        base_score = base_scaled_scores[i]

        # Startin with base score
        score = base_score

        # Diminishing Returns for clicked categories
        cat_oids = article.get("category", [])
        article_lang = article.get("language", "english").lower()

        # Category Bonus & Tag Bonus are accumulated separately
        cat_bonus = 0.0
        for cdict in cat_oids:
            oid = cdict.get("$oid")
            if oid and category_map:
                cat_name = category_map.get(oid, "unknown")
                user_clicks = custom_cat_clicks.get(cat_name, 0)

                # Diminishing returns formula
                cat_bonus += alpha * (1.0 / (1.0 + user_clicks / k))

        # Diminishing Returns for clicked tags
        tag_bonus = 0.0
        article_tags = article.get("tags", [])
        for t in article_tags:

            # if we had a tag map, we'd convert t -> tag_name (currently this is not handled)
            # or if t is direct string -> just use t
            if isinstance(t, str):
                user_tag_clicks = custom_tag_clicks.get(t, 0)
                tag_bonus += alpha * (1.0 / (1.0 + user_tag_clicks / k))

        # Combining the category and tag bonus scores
        score += (cat_bonus + tag_bonus)

        # Novelty Boost (for near-zero clicks, which indicates that the user has rarely interacted with this type of article)
        if article_lang == user_lang:
            novelty_sum = 0.0
            for cdict in cat_oids:
                oid = cdict.get("$oid")
                if oid and category_map:
                    cat_name = category_map.get(oid, "unknown")
                    user_clicks = custom_cat_clicks.get(cat_name, 0)

                    # If user_clicks < 2 => big novelty boost
                    # If user_clicks=1   => half novelty boost
                    # If user_clicks=0   => full novelty boost
                    # and so on...
                    if user_clicks < 2:
                        factor = max(0.0, (2 - user_clicks) / 2.0)  # range [1..0.5..0]
                        novelty_sum += (novelty_boost * factor)

            # Also checking tags for including in the novel boost
            for t in article_tags:
                if isinstance(t, str):
                    user_tag_clicks = custom_tag_clicks.get(t, 0)
                    if user_tag_clicks < 2:
                        factor = max(0.0, (2 - user_tag_clicks) / 2.0)
                        novelty_sum += (novelty_boost * factor)

            score += novelty_sum

        final_scores.append(score)

    # Re-scale final scores to between 0 and 100 as integers in descending order
    final_scaled_scores = min_max_scale(final_scores)
    ranked = sorted(enumerate(final_scaled_scores), key=lambda x: x[1], reverse=True)
    return ranked


def min_max_scale(values):
    arr = np.array(values, dtype=float)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr, dtype=int) * 50
    scaled = (arr - mn) / (mx - mn) * 100.0
    return np.round(scaled).astype(int)
