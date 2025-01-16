import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Ensure nltk resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

def assign_cohorts(users, user_preferences, category_map, n_clusters=5):
    """
    Dynamically assigns cohorts to all users via clustering on their
    preferred categories (converted to text).
    
    Returned dictionary:
        user_cohort_map: dict { user['_id'] -> "cohort_label" }
    """
    # Building map user_id -> user_preferences doc
    prefs_map = {str(up["user_id"]): up for up in user_preferences}

    # Fetch stopwords from nltk
    stop_words = set(stopwords.words("english"))

    # Creating a "document" by joining all preferred category names for each user
    user_docs = []
    user_ids = []
    for user in users:
        uid = str(user["_id"])
        user_pref = prefs_map.get(uid, {})
        cat_ids = user_pref.get("article_category", [])  # list of category OIDs
        
        # Convert each category OID to a readable name
        category_names = []
        for cid in cat_ids:
            cat_name = category_map.get(cid, "")
            if cat_name:
                category_names.append(cat_name.lower())
        
        # Joining all the category names into a single string "document"
        doc_text = " ".join(category_names) if category_names else "none"

        # Tokenize and remove stopwords using nltk
        tokens = word_tokenize(doc_text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        filtered_doc = " ".join(filtered_tokens)

        user_docs.append(filtered_doc)
        user_ids.append(uid)

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(user_docs)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Identifying top words for each cluster
    top_words_per_cluster = _get_top_words(kmeans, vectorizer, top_n=2)

    # Building a map: user_id -> "word1-word2" for that cluster
    user_cohort_map = {}
    for i, uid in enumerate(user_ids):
        cluster_id = clusters[i]
        user_cohort_map[uid] = top_words_per_cluster[cluster_id]

    return user_cohort_map

def assign_cohort_to_custom_user(custom_user, kmeans, vectorizer, top_words_per_cluster):
    """
    Assigns a cohort to the custom user based on the clusters formed.
    """
    stop_words = set(stopwords.words("english"))
    # Prepare the custom user's document
    category_names = [cat[0].lower() for cat in custom_user["favorite_categories"]]
    tag_names = [tag[0].lower() for tag in custom_user["favorite_tags"]]
    doc_text = " ".join(category_names + tag_names)
    tokens = word_tokenize(doc_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_doc = " ".join(filtered_tokens)

    # Vectorize the custom user's document
    custom_user_vector = vectorizer.transform([filtered_doc])

    # Predict the cluster for the custom user
    cluster_id = kmeans.predict(custom_user_vector)[0]
    cohort_label = top_words_per_cluster[cluster_id]
    
    return cohort_label

def _get_top_words(kmeans_model, vectorizer, top_n=2):
    """
    For each cluster, returns a string made of the top_n words 
    with highest TF-IDF weight in that cluster's centroid.
    """
    centers = kmeans_model.cluster_centers_  # shape: (n_clusters, n_features)
    feature_names = vectorizer.get_feature_names_out()
    
    top_words_list = []
    for cluster_id in range(len(centers)):
        center = centers[cluster_id]
        top_indices = center.argsort()[::-1][:top_n]
        top_words = [feature_names[idx] for idx in top_indices]
        top_words_list.append("-".join(top_words))

    return top_words_list
