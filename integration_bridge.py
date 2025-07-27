import sys
import pandas as pd
from pathlib import Path
import json
import os
import re
from transformers import pipeline
from rapidfuzz import fuzz

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.loaders import ConversationLoader
from src.models.profile_generator import ProfileGenerator
from src.features.behavioral_analyzer import BehavioralAnalyzer

class LegacyAnalysisIntegrator:
    """Intègre votre analyse existante avec le nouveau système"""

    def __init__(self):
        self.profile_generator = ProfileGenerator()
        self.behavioral_analyzer = BehavioralAnalyzer()

    # Vos fonctions originales adaptées
    def clean_text(self, text):
        """Votre fonction de nettoyage originale"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'[^\w\s\?\!\.\,]', '', text)
        return text

    def detect_questions(self, df):
        """Votre détection de questions originale"""
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

    def detect_hesitations(self, df):
        """Votre détection d'hésitations originale"""
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

    def analyze_sentiment_legacy(self, df):
        """Votre analyse de sentiment originale"""
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

        texts = df['user_input_clean'].tolist()
        results = sentiment_analyzer(texts)
        df['sentiment_label'] = [res['label'] for res in results]
        df['sentiment_score'] = [int(res['label'].split()[0]) for res in results]  # Extract numeric score
        return df

    def detect_repeated_questions(self, df, threshold=90):
        """Votre détection de questions répétées originale"""
        repeated_flags = [False] * len(df)
        questions = df['user_input_clean'].tolist()

        for i in range(len(questions)):
            for j in range(i+1, len(questions)):
                if "?" in questions[i] or "?" in questions[j]:
                    similarity = fuzz.ratio(questions[i], questions[j])
                    if similarity >= threshold:
                        repeated_flags[j] = True

        df['is_repeated_question'] = repeated_flags
        return df

    def evaluate_comprehension(self, df):
        """Votre évaluation de compréhension originale"""
        df_questions = df[df['is_question'] == True]
        if df_questions.empty:
            return pd.DataFrame(columns=['session_id', 'total_questions', 'repeated_questions', 'repetition_rate', 'comprehension_score'])

        stats = df_questions.groupby('session_id').agg(
            total_questions=('user_input_clean', 'count'),
            repeated_questions=('is_repeated_question', 'sum')
        ).reset_index()

        stats['repetition_rate'] = stats['repeated_questions'] / stats['total_questions'].replace(0, 1)

        def score(row):
            if row['repetition_rate'] < 0.2:
                return 1  # Bon
            elif row['repetition_rate'] <= 0.5:
                return 0.5  # Moyen
            else:
                return 0  # Faible

        stats['comprehension_score'] = stats.apply(score, axis=1)
        return stats

    def run_legacy_analysis(self, conversations_folder: str, output_folder: str = "output_legacy"):
        """Exécute votre analyse originale avec les améliorations"""
        print("🔄 Exécution de l'analyse avec votre logique originale...")

        # Chargement avec le nouveau loader
        loader = ConversationLoader(conversations_folder)
        conversations = loader.load_conversations()
        df = loader.to_dataframe(conversations)

        if df.empty:
            print("❌ Aucune conversation trouvée")
            return

        print(f"✅ Chargé {len(df)} conversations de {df['session_id'].nunique()} sessions")

        # Appliquer votre pipeline original
        print("🧹 Nettoyage des textes...")
        df['user_input_clean'] = df['user_input'].apply(self.clean_text)
        df['bot_response_clean'] = df['bot_response'].apply(self.clean_text)

        print("❓ Détection des questions...")
        df = self.detect_questions(df)

        print("⏱️ Calcul des écarts temporels...")
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['session_id', 'timestamp_dt'])
        df['time_diff_sec'] = df.groupby('session_id')['timestamp_dt'].diff().dt.total_seconds()
        df['time_diff_sec'] = df['time_diff_sec'].fillna(0)

        print("🤔 Détection des hésitations...")
        df = self.detect_hesitations(df)

        print("😊 Analyse des sentiments...")
        df = self.analyze_sentiment_legacy(df)

        print("📏 Calcul des longueurs de messages...")
        df['user_input_len'] = df['user_input_clean'].apply(lambda x: len(x.split()))
        df['bot_response_len'] = df['bot_response_clean'].apply(lambda x: len(x.split()))

        print("🔄 Détection des questions répétées...")
        df = self.detect_repeated_questions(df)

        print("🧠 Évaluation de la compréhension...")
        comprehension_stats = self.evaluate_comprehension(df)

        # Créer le dossier de sortie
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les résultats originaux
        comprehension_stats.to_csv(output_path / 'comprehension_scores.csv', index=False)

        # Agrégation des features (votre logique originale)
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

        # Étiquetage tonalité (votre logique)
        def sentiment_label(score):
            if score >= 3.5:
                return "positif"
            elif score <= 2.5:
                return "négatif"
            else:
                return "neutre"

        agg['tonalite_globale'] = agg['sentiment_moyen_session'].apply(sentiment_label)

        # Détection sessions difficiles (votre logique)
        def detect_sessions_difficiles(row):
            return (row['tonalite_globale'] == 'négatif' and
                   row['taux_hesitation'] > 0.3 and
                   row['score_comprehension_moyen'] < 0.5)

        agg['session_difficile'] = agg.apply(detect_sessions_difficiles, axis=1)

        # Sauvegarder features_sessions.csv (comme votre version originale)
        agg.to_csv(output_path / 'features_sessions.csv', index=False)

        print(f"✅ Analyse terminée ! Résultats dans {output_path}")
        print(f"📊 {len(agg)} sessions analysées")
        print(f"🚨 {agg['session_difficile'].sum()} sessions difficiles détectées")

        # BONUS: Générer aussi les profils avancés
        print("\n🆕 Génération des profils avancés (nouveau système)...")

        for session_id in df['session_id'].unique():
            session_data = df[df['session_id'] == session_id]
            try:
                profile = self.profile_generator.generate_profile(session_data)

                # Sauvegarder le profil avancé
                profile_json = self.profile_generator.export_profile(session_id, 'json')
                with open(output_path / f"{session_id}_advanced_profile.json", 'w', encoding='utf-8') as f:
                    f.write(profile_json)

            except Exception as e:
                print(f"⚠️ Erreur profil {session_id}: {e}")

        print("✨ Profils avancés générés !")
        return agg

def main():
    """Interface en ligne de commande"""
    import argparse

    parser = argparse.ArgumentParser(description="Bridge entre votre analyse existante et le nouveau système")
    parser.add_argument('--conversations-folder', required=True, help='Dossier contenant vos fichiers JSON de conversations')
    parser.add_argument('--output-folder', default='output_integrated', help='Dossier de sortie')

    args = parser.parse_args()

    integrator = LegacyAnalysisIntegrator()
    results = integrator.run_legacy_analysis(args.conversations_folder, args.output_folder)

    print("\n📈 Aperçu des résultats:")
    print(results.head(10))

if __name__ == "__main__":
    main()
