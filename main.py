import os
import json
import pandas as pd
import re
from transformers import pipeline
from rapidfuzz import fuzz


# ----------- Étape 1 : Chargement des données JSON -----------

def load_conversations(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            session_id = filename.split('.')[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
                for exchange in conversations:
                    exchange['session_id'] = session_id
                    all_data.append(exchange)
    df = pd.DataFrame(all_data)
    return df

# ----------- Étape 2 : Nettoyage du texte -----------

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s\?\!\.\,]', '', text)  # garder ponctuation utile
    return text

def preprocess_text(df):
    df['user_input_clean'] = df['user_input'].apply(clean_text)
    df['bot_response_clean'] = df['bot_response'].apply(clean_text)
    return df

# ----------- Fonction sauvegarde conversations nettoyées -----------

def save_cleaned_conversations(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    grouped = df.groupby('session_id')
    
    for session_id, group in grouped:
        cleaned_exchanges = []
        for _, row in group.iterrows():
            cleaned_exchanges.append({
                "timestamp": row['timestamp'],
                "user_input": row['user_input_clean'],
                "bot_response": row['bot_response_clean']
            })
        output_path = os.path.join(output_folder, f"{session_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_exchanges, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Conversations nettoyées sauvegardées dans : {output_folder}")

# ----------- Étape 3 : Détection des questions -----------

def detect_questions(df):
    interrogatives = ['quoi', 'comment', 'pourquoi', 'qui', 'où', 'quand', 'est-ce que', 'quel', 'quelle', '?']

    def check_question(text):
        if pd.isna(text):
            return False
        text = text.lower()
        if '?' in text:
            return True
        for word in interrogatives:
            if word in text:
                return True
        return False

    df['is_question'] = df['user_input_clean'].apply(check_question)
    return df

# ----------- Étape 4 : Écarts de temps entre messages -----------

def add_time_differences(df):
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['session_id', 'timestamp_dt'])
    df['time_diff_sec'] = df.groupby('session_id')['timestamp_dt'].diff().dt.total_seconds()
    df['time_diff_sec'] = df['time_diff_sec'].fillna(0)
    return df

# ----------- Étape 5 : Détection des hésitations -----------

def detect_hesitations(df):
    hesitations = ['euh', 'mmh', 'hmm', r'\.\.\.']

    def has_hesitation(text):
        if pd.isna(text):
            return False
        text = text.lower()
        for h in hesitations:
            if re.search(h, text):
                return True
        return False

    df['has_hesitation'] = df['user_input_clean'].apply(has_hesitation)
    return df

# ----------- Étape 6 : Analyse des sentiments -----------

def analyze_sentiment(df):
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    
    texts = df['user_input_clean'].tolist()
    results = sentiment_analyzer(texts)
    df['sentiment_label'] = [res['label'] for res in results]
    df['sentiment_score'] = [res['score'] for res in results]
    return df

# ----------- Étape 7 : Longueur des messages -----------

def add_text_lengths(df):
    df['user_input_len'] = df['user_input_clean'].apply(lambda x: len(x.split()))
    df['bot_response_len'] = df['bot_response_clean'].apply(lambda x: len(x.split()))
    return df

# ----------- Étape 8 : Détection des questions répétées -----------

def detect_repeated_questions(df, threshold=90):
    repeated_flags = [False] * len(df)
    questions = df['user_input_clean'].tolist()

    for i in range(len(questions)):
        for j in range(i+1, len(questions)):
            if questions[i].endswith("?") or questions[j].endswith("?"):
                similarity = fuzz.ratio(questions[i], questions[j])
                if similarity >= threshold:
                    repeated_flags[j] = True

    df['is_repeated_question'] = repeated_flags
    return df

# ----------- Étape 9 : Évaluation du niveau de compréhension -----------

def evaluate_comprehension(df):
    df_questions = df[df['is_question'] == True]
    stats = df_questions.groupby('session_id').agg(
        total_questions=('user_input_clean', 'count'),
        repeated_questions=('is_repeated_question', 'sum')
    ).reset_index()
    stats['repetition_rate'] = stats['repeated_questions'] / stats['total_questions']

    def score(row):
        if row['repetition_rate'] < 0.2:
            return 1  # Bon
        elif row['repetition_rate'] <= 0.5:
            return 0.5  # Moyen
        else:
            return 0  # Faible

    stats['comprehension_score'] = stats.apply(score, axis=1)
    return stats

# ----------- Étape 10 : Agrégation des features par session -----------

def aggregate_features(df, comprehension_stats):
    df_merged = df.merge(comprehension_stats[['session_id', 'comprehension_score']], on='session_id', how='left')
    agg = df_merged.groupby('session_id').agg({
        'sentiment_score': 'mean',
        'time_diff_sec': 'mean',
        'user_input': 'count',
        'has_hesitation': 'mean',
        'is_question': 'sum',
        'user_input_len': 'mean',
        'comprehension_score': 'mean'
    }).reset_index()

    agg.rename(columns={
        'sentiment_score': 'sentiment_moyen_session',
        'time_diff_sec': 'delai_moyen_sec',
        'user_input': 'nb_echanges',
        'has_hesitation': 'taux_hesitation',
        'is_question': 'nb_questions',
        'user_input_len': 'nb_mots_moy',
        'comprehension_score': 'score_comprehension_moyen'
    }, inplace=True)
    return agg

# ----------- Étape 11 : Étiquetage tonalité globale -----------

def sentiment_label(score):
    if score >= 3.5:
        return "positif"
    elif score <= 2.5:
        return "négatif"
    else:
        return "neutre"

# ----------- Étape 12 : Détection sessions difficiles -----------

def detect_sessions_difficiles(row):
    if (row['tonalite_globale'] == 'négatif' and
        row['taux_hesitation'] > 0.3 and
        row['score_comprehension_moyen'] < 0.5):
        return True
    else:
        return False

# ----------- Programme principal -----------

if __name__ == "__main__":
    folder = r'C:\Users\21696\Downloads\conversations\conversations'

    print("Chargement des conversations...")
    df = load_conversations(folder)
    
    print("Nettoyage des textes...")
    df = preprocess_text(df)
    
    # Sauvegarde des fichiers JSON nettoyés
    cleaned_folder = r'C:\Users\21696\Downloads\conversations\conversations_cleaned'
    save_cleaned_conversations(df, cleaned_folder)
    
    print("Détection des questions...")
    df = detect_questions(df)
    
    print("Calcul des écarts temporels...")
    df = add_time_differences(df)
    
    print("Détection des hésitations...")
    df = detect_hesitations(df)
    
    print("Analyse des sentiments...")
    df = analyze_sentiment(df)
    
    print("Calcul des longueurs de messages...")
    df = add_text_lengths(df)
    
    print("Détection des questions répétées...")
    df = detect_repeated_questions(df)
    
    print("Évaluation de la compréhension...")
    stats = evaluate_comprehension(df)
    stats.to_csv('comprehension_scores.csv', index=False)
    print("Scores de compréhension sauvegardés dans 'comprehension_scores.csv'.")
    
    print("Agrégation des features par session...")
    df_features = aggregate_features(df, stats)
    df_features['tonalite_globale'] = df_features['sentiment_moyen_session'].apply(sentiment_label)
    
    print("Détection des sessions difficiles...")
    df_features['session_difficile'] = df_features.apply(detect_sessions_difficiles, axis=1)
    
    df_features.to_csv('features_sessions.csv', index=False)
    print("Fichier 'features_sessions.csv' généré avec succès.")
    
    # Affichage d'exemple
    print(df_features.head(10))
