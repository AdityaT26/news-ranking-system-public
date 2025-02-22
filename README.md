﻿# news-ranking-system
# Personalized News Ranking System

This project is a Python-based machine learning system that ranks and personalizes news articles for users based on their preferences and behavioral data. It uses clustering and Natural Language Processing (NLP) techniques to dynamically adapt to user preferences and assigns cohorts based on shared interests.

## Features

- **Dynamic Cohort Assignment**: Users are grouped into cohorts using clustering based on their preferred categories.
- **Article Ranking**: Ranks articles for each user based on their preferences and assigns relevant tags.
- **Tagging**: Automatically assigns a descriptive tag to each article using NLP techniques.
- **Preprocessing**: Handles stopword removal, tokenization, and TF-IDF vectorization.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/news-ranking-system.git
   cd news-ranking-system
   ```

2. **Set Up the Environment**:
   It’s recommended to use a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate    # On Windows: env\\Scripts\\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Resources**:
   The project uses `nltk` for tokenization and stopword removal:
   ```bash
   python -m nltk.downloader punkt stopwords
   ```

---

## Running the Project

### 1. Assigning Cohorts and Ranking Articles
Run the main script to process user data, assign cohorts, and rank articles:
```bash
python main.py --mode <mode> --local
```
- **mode**: *training* for building the feature matrix and training the xgboost regressor.
- **mode**: *production* for testing the program.
- **--local**: Use this argument for training or testing using the local JSON files.

### 2. Sample Output
The system will output:
- **User Cohorts**: Each user is assigned a cohort label based on shared preferences.
- **Ranked Articles**: Top-ranked articles for users, including dynamically assigned tags.

---

## Example Use Case

1. **Input**:
   - User preferences from `user_preferences.json`
   - Articles from `articles.json`
   - Category mappings from `article_categories.json`

2. **Output**:
   - User `12345` is assigned to cohort `"technology-ai"`.
   - Articles are tagged and ranked for relevance:
     ```
     [95/100] "AI Revolution in Healthcare" (Tag: technology)
     [90/100] "Advancements in Renewable Energy" (Tag: energy)
     ```


