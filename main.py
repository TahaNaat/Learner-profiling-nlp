import os
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import config
from src.data.loaders import ConversationLoader, QuestionnaireLoader
from src.features.behavioral_analyzer import BehavioralAnalyzer
from src.models.profile_generator import ProfileGenerator
from src.visualization.dashboard import LearnerDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('learner_profiling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LearnerProfilingPipeline:
    """Main pipeline for learner profiling and analysis"""

    def __init__(self):
        # Use the configured data directory directly
        self.conversation_loader = ConversationLoader(config.data_dir)
        self.questionnaire_loader = QuestionnaireLoader(config.data_dir / "questionnaires") if (config.data_dir / "questionnaires").exists() else None

        # Import with error handling
        try:
            from src.features.behavioral_analyzer import BehavioralAnalyzer
            self.behavioral_analyzer = BehavioralAnalyzer()
        except Exception as e:
            logger.warning(f"Behavioral analyzer initialization failed: {e}")
            self.behavioral_analyzer = None

        self.profile_generator = ProfileGenerator()

        # Ensure output directories exist
        try:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / "profiles").mkdir(exist_ok=True)
            (config.output_dir / "analytics").mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create output directories: {e}")

    def run_full_pipeline(self):
        """Run the complete analysis pipeline"""
        logger.info("Démarrage du pipeline d'analyse des apprenants")

        try:
            # Step 1: Load conversation data
            logger.info("Étape 1: Chargement des données de conversation")
            conversations = self.conversation_loader.load_conversations()
            if not conversations:
                logger.error("Aucune conversation trouvée. Arrêt du pipeline.")
                return

            df_conversations = self.conversation_loader.to_dataframe(conversations)
            logger.info(f"Chargé {len(conversations)} conversations de {df_conversations['session_id'].nunique()} apprenants")

            # Step 2: Tentative de chargement des données de questionnaires (optionnel)
            logger.info("Étape 2: Vérification des données de questionnaires (optionnel)")
            questionnaire_data = pd.DataFrame()  # Pas de questionnaires dans votre cas

            if self.questionnaire_loader:
                try:
                    vark_data = self.questionnaire_loader.load_vark_responses()
                    mbti_data = self.questionnaire_loader.load_mbti_responses()

                    # Combine questionnaire data if available
                    if not vark_data.empty:
                        vark_data['questionnaire_type'] = 'VARK'
                        questionnaire_data = pd.concat([questionnaire_data, vark_data], ignore_index=True)
                    if not mbti_data.empty:
                        mbti_data['questionnaire_type'] = 'MBTI'
                        questionnaire_data = pd.concat([questionnaire_data, mbti_data], ignore_index=True)

                    if not questionnaire_data.empty:
                        logger.info(f"Questionnaires trouvés: {len(questionnaire_data)} réponses")
                    else:
                        logger.info("Aucun questionnaire trouvé - utilisation des conversations uniquement")
                except Exception as e:
                    logger.info(f"Pas de questionnaires disponibles - analyse basée sur conversations uniquement: {e}")
            else:
                logger.info("Module questionnaire non disponible - analyse basée sur conversations uniquement")

            # Step 3: Behavioral analysis per session
            logger.info("Étape 3: Analyse comportementale par session")
            behavioral_features = {}
            session_ids = df_conversations['session_id'].unique()

            if self.behavioral_analyzer:
                for session_id in session_ids:
                    session_data = df_conversations[df_conversations['session_id'] == session_id]
                    try:
                        features = self.behavioral_analyzer.analyze_session_behavior(session_data)
                        behavioral_features[session_id] = features
                        logger.info(f"Analyse terminée pour la session {session_id}")
                    except Exception as e:
                        logger.warning(f"Erreur analyse comportementale pour {session_id}: {e}")
            else:
                logger.warning("Analyseur comportemental non disponible - analyse de base uniquement")

            # Step 4: Profile generation
            logger.info("Étape 4: Génération des profils d'apprenants")
            profiles = {}

            for session_id in session_ids:
                session_data = df_conversations[df_conversations['session_id'] == session_id]
                session_questionnaire = questionnaire_data[
                    questionnaire_data.get('learner_id', pd.Series()) == session_id
                ] if not questionnaire_data.empty else None

                profile = self.profile_generator.generate_profile(
                    session_data, 
                    session_questionnaire,
                    [behavioral_features.get(session_id)]
                )
                profiles[session_id] = profile
                logger.info(f"Profil généré pour {session_id}")

            # Step 5: Save results
            logger.info("Étape 5: Sauvegarde des résultats")
            self._save_results(profiles, behavioral_features, df_conversations)

            # Step 6: Generate analytics dashboard
            logger.info("Étape 6: Génération du tableau de bord")
            self._create_dashboard(df_conversations)

            logger.info("Pipeline terminé avec succès")

        except Exception as e:
            logger.error(f"Erreur dans le pipeline: {str(e)}")
            raise

    def _save_results(self, profiles, behavioral_features, conversations_df):
        """Save analysis results to files"""

        # Save individual profiles
        for learner_id, profile in profiles.items():
            profile_json = self.profile_generator.export_profile(learner_id, 'json')
            with open(config.output_dir / "profiles" / f"{learner_id}_profile.json", 'w', encoding='utf-8') as f:
                f.write(profile_json)

        # Save consolidated profile summary
        profile_summary = []
        for learner_id, profile in profiles.items():
            profile_summary.append({
                'learner_id': profile.learner_id,
                'visual_learning': profile.learning_style.visual,
                'auditory_learning': profile.learning_style.auditory,
                'reading_learning': profile.learning_style.reading,
                'kinesthetic_learning': profile.learning_style.kinesthetic,
                'extraversion': profile.personality.extraversion,
                'intuition': profile.personality.intuition,
                'thinking': profile.personality.thinking,
                'judging': profile.personality.judging,
                'engagement_level': profile.behavioral.engagement_level,
                'persistence': profile.behavioral.persistence,
                'help_seeking': profile.behavioral.help_seeking,
                'confidence_level': profile.confidence_level,
                'num_strengths': len(profile.strengths),
                'num_difficulties': len(profile.difficulty_areas),
                'num_recommendations': len(profile.recommended_strategies),
                'last_updated': profile.last_updated
            })

        profile_summary_df = pd.DataFrame(profile_summary)
        profile_summary_df.to_csv(config.output_dir / "analytics" / "profile_summary.csv", index=False, encoding='utf-8')

        # Save behavioral features summary
        behavioral_summary = []
        for session_id, features in behavioral_features.items():
            behavioral_summary.append({
                'session_id': features.session_id,
                'hesitation_rate': features.hesitation_rate,
                'question_complexity': features.question_complexity,
                'engagement_score': features.engagement_score,
                'learning_pace': features.learning_pace,
                'confusion_indicators': features.confusion_indicators,
                'num_semantic_clusters': len(set(features.semantic_clusters)) if features.semantic_clusters else 0
            })

        behavioral_summary_df = pd.DataFrame(behavioral_summary)
        behavioral_summary_df.to_csv(config.output_dir / "analytics" / "behavioral_features.csv", index=False, encoding='utf-8')

        # Save detailed conversation analysis
        conversations_df.to_csv(config.output_dir / "analytics" / "processed_conversations.csv", index=False, encoding='utf-8')

        logger.info(f"Résultats sauvegardés dans {config.output_dir}")

    def _create_dashboard(self, conversations_df):
        """Create and optionally launch the analytics dashboard"""
        try:
            dashboard = LearnerDashboard(self.profile_generator, conversations_df)

            # Save dashboard as HTML (optional)
            # dashboard.app.run_server(debug=False, port=8050, host='127.0.0.1')

            logger.info("Tableau de bord créé. Lancez dashboard.py pour l'interface interactive.")

        except Exception as e:
            logger.warning(f"Impossible de créer le tableau de bord: {str(e)}")

    def run_realtime_analysis(self, session_id: str):
        """Run analysis for a specific session in real-time"""
        logger.info(f"Analyse en temps réel pour la session {session_id}")

        try:
            # Load only the specific session data
            conversations = self.conversation_loader.load_conversations()
            session_conversations = [conv for conv in conversations if conv.session_id == session_id]

            if not session_conversations:
                logger.error(f"Aucune conversation trouvée pour la session {session_id}")
                return None

            df_session = self.conversation_loader.to_dataframe(session_conversations)

            # Perform behavioral analysis
            behavioral_features = self.behavioral_analyzer.analyze_session_behavior(df_session)

            # Generate or update profile
            if session_id in self.profile_generator.profile_cache:
                profile = self.profile_generator.update_profile(session_id, df_session)
            else:
                profile = self.profile_generator.generate_profile(df_session)

            logger.info(f"Analyse temps réel terminée pour {session_id}")
            return profile, behavioral_features

        except Exception as e:
            logger.error(f"Erreur dans l'analyse temps réel: {str(e)}")
            return None

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Système d'Analyse et de Profilage des Apprenants")
    parser.add_argument('--mode', choices=['full', 'realtime', 'dashboard', 'api'], 
                       default='full', help='Mode d\'exécution')
    parser.add_argument('--session-id', help='ID de session pour l\'analyse temps réel')
    parser.add_argument('--port', type=int, default=8050, help='Port pour le dashboard ou l\'API')
    parser.add_argument('--data-dir', help='Répertoire des données d\'entrée')
    parser.add_argument('--output-dir', help='Répertoire de sortie')

    args = parser.parse_args()

    # Override config with command line arguments if provided
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    # Initialize pipeline
    pipeline = LearnerProfilingPipeline()

    if args.mode == 'full':
        # Run complete analysis pipeline
        pipeline.run_full_pipeline()

    elif args.mode == 'realtime':
        # Run real-time analysis for specific session
        if not args.session_id:
            logger.error("--session-id requis pour le mode temps réel")
            return

        result = pipeline.run_realtime_analysis(args.session_id)
        if result:
            profile, behavioral = result
            print(f"\nProfil mis à jour pour {profile.learner_id}")
            print(f"Niveau de confiance: {profile.confidence_level:.2%}")
            print(f"Recommandations: {len(profile.recommended_strategies)}")

    elif args.mode == 'dashboard':
        # Launch interactive dashboard
        try:
            conversations = pipeline.conversation_loader.load_conversations()
            df = pipeline.conversation_loader.to_dataframe(conversations)
            dashboard = LearnerDashboard(pipeline.profile_generator, df)
            print(f"Lancement du tableau de bord sur http://localhost:{args.port}")
            dashboard.run(debug=False, port=args.port)
        except Exception as e:
            logger.error(f"Erreur lors du lancement du dashboard: {str(e)}")

    elif args.mode == 'api':
        # Launch REST API server
        try:
            import uvicorn
            from src.api.endpoints import app
            print(f"Lancement de l'API sur http://localhost:{args.port}")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        except Exception as e:
            logger.error(f"Erreur lors du lancement de l'API: {str(e)}")

if __name__ == "__main__":
    main()
