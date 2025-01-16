from data_loader import load_data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import json
import random
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def conv_todict(lis):
    merged_dict = {}
    for item in lis:
        for key, value in item.items():
            if key not in merged_dict:
                merged_dict[key] = value
    return merged_dict

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text)

def make_local_users(use_local_json=True):
    data_dict = load_data(use_local_json=use_local_json, db=None)

    article_categories = data_dict["article_categories"]
    article_categories = conv_todict([{cat["_id"]: cat["name"]} for cat in article_categories])

    article_file = "articles.json"
    articles = {}

    if os.path.exists(article_file) and use_local_json:
        print("Loading articles from cached file...")
        with open(article_file, 'r') as f:
            articles = json.load(f)
    else:
        articles = data_dict["articles"]
        articles = conv_todict([{art["_id"]["$oid"] : [art["language"], art["title"], [cat["$oid"] for cat in art["category"]][0], art["body"] ]} for art in articles if art["body"]])

        articles_updated = {}

        for art_no, art in articles.items():
            try:
                art[2] = article_categories[art[2]]
                articles_updated.update({art_no: art})
            except KeyError:
                pass

        articles = articles_updated

        tag_options = ["accidents","cricket","awards and recognitions","human rights","crime","politics","education","natural disasters","economy","climate and weather","elections","celebrity","movies","supply chain and logistics","financial markets","religious events and festivals","government","pharma and healthcare","sports","energy","conflicts & war","telecom","diseases","automotive","research","mental health","wildlife","corporate news","social media and internet","ocean conservation","startups & entrepreneurship","eco-friendly","pollution","banking and finance","soccer","entertainment","fmcg","tourism","television","food","technology","health and fitness","national security","insurance","real estate","cybercrime and cybersecurity","fashion and lifestyle","artificial intelligence","wrestling","gaming","international trade","e-commerce","home and interior design","cryptocurrencies","american football","baseball","basketball","law and justice","golf","religion","recycling","metal & mining","nonprofit organizations","corporate social responsibility","science and innovations","agriculture and farming","space","mixed martial arts","tennis","motorsports","renewable energy","aviation","terrorism","lgbtq","boxing","field hockey","volleyball","textile","immigration and migrant issues","rugby","chemicals","work-life balance","philanthropy"]
        
        cleaned_tag_options = [clean_text(tag) for tag in tag_options]
        vectorizer = TfidfVectorizer()
        tag_matrix = vectorizer.fit_transform(cleaned_tag_options)

        for art_no, art in articles.items():
            title = art[1]
            body = art[3]
            text = title + " " + body
            cleaned_text = clean_text(text)
            text_vector = vectorizer.transform([cleaned_text])
            similarities = cosine_similarity(text_vector, tag_matrix)
            most_similar_index = similarities.argmax()
            articles[art_no].append(tag_options[most_similar_index])

        with open(article_file, 'w') as f:
            json.dump(articles, f, indent=4)

    return articles

def get_user_preferences():
    while True:
        language = input("Enter your preferred language (english/hindi): ").lower()
        if language in ["english", "hindi"]:
            break
        else:
            print("Invalid language. Please enter 'english' or 'hindi'.")

    while True:
        gender = input("Enter your gender (male/female): ").lower()
        if gender in ["male", "female"]:
            break
        else:
            print("Invalid gender. Please enter 'male' or 'female'.")
    return language, gender

def article_interaction(articles, language):
    favorite_categories = []
    favorite_tags = []
    displayed_article_ids = set()

    while True:
        print("\n--- Available Articles ---")
        available_articles = [
            (art_id, art)
            for art_id, art in articles.items() if art[0] == language and art_id not in displayed_article_ids
        ]

        if not available_articles:
            print("No more new articles found in your preferred language.")
            break

        if len(available_articles) < 5:
          num_to_display = len(available_articles)
        else:
          num_to_display = 5

        displayed_articles_subset = random.sample(available_articles, num_to_display)

        for i, (art_id, article) in enumerate(displayed_articles_subset):
            print(f"{i + 1}. Title: {article[1]}")
            print(f"   Category: {article[2]}")
            print(f"   Tag: {article[4]}")
            print("-" * 20)
            displayed_article_ids.add(art_id)

        choice = input("Enter the number of the article you're most interested in (or press Enter to skip/ 'q' to quit): ")

        if choice.lower() == 'q':
            break

        if not choice:
            print("Skipping to the next set of articles.")
            continue

        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(displayed_articles_subset):
                chosen_art_id, chosen_article = displayed_articles_subset[choice_index]
                favorite_categories.append(chosen_article[2])
                favorite_tags.append(chosen_article[4])
                print(f"You chose: {chosen_article[1]}")
            else:
                print("Invalid article number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

    return favorite_categories, favorite_tags

def save_user_data(user_data, user_data_freq):
    user_file = "user_data.json"
    user_file_freq = "user_data_freq.json"
    with open(user_file, 'w') as f:
        json.dump(user_data, f, indent=4)
    print(f"User data saved to {user_file}")

    with open(user_file_freq, 'w') as f:
        json.dump(user_data_freq, f, indent=4)
    print(f"User data with frequencies saved to {user_file_freq}")


def main():
    articles = make_local_users()
    language, gender = get_user_preferences()
    favorite_categories, favorite_tags = article_interaction(articles, language)

    # Count the occurrences of each category and tag
    top_categories_with_counts = Counter(favorite_categories).most_common(10)
    top_tags_with_counts = Counter(favorite_tags).most_common(10)

    # Extract the category and tag names from the Counter output
    top_categories = [category for category, count in top_categories_with_counts]
    top_tags = [tag for tag, count in top_tags_with_counts]

    user_data = {
        "language": language,
        "gender": gender,
        "favorite_categories": top_categories,
        "favorite_tags": top_tags
    }

    user_data_freq = {
        "language": language,
        "gender": gender,
        "favorite_categories": top_categories_with_counts,
        "favorite_tags": top_tags_with_counts
    }
    save_user_data(user_data,user_data_freq)

if __name__ == "__main__":
    main()