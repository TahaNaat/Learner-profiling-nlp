import sys
import json
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def create_test_data():
    """Créer des données de test"""
    test_conversations = [
        {
            "user_input": "Bonjour, comment ça marche?",
            "bot_response": "Bonjour! Je vais vous expliquer...",
            "timestamp": "2024-01-01T10:00:00"
        },
        {
            "user_input": "Je ne comprends pas cette partie",
            "bot_response": "Permettez-moi de clarifier...",
            "timestamp": "2024-01-01T10:01:00"
        },
        {
            "user_input": "Pouvez-vous donner un exemple visuel?",
            "bot_response": "Voici un diagramme...",
            "timestamp": "2024-01-01T10:02:00"
        },
        {
            "user_input": "Parfait, je comprends mieux maintenant!",
            "bot_response": "Excellent! Avez-vous d'autres questions?",
            "timestamp": "2024-01-01T10:03:00"
        }
    ]
    return test_conversations

def test_data_loading():
    """Test du chargement des données"""
    print("🧪 Test 1: Chargement des données")

    # Créer un dossier temporaire avec des données de test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Créer un fichier de conversation test
        test_data = create_test_data()
        with open(temp_path / "test_session.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        # Test du loader
        try:
            from src.data.loaders import ConversationLoader
            loader = ConversationLoader(temp_path)
            conversations = loader.load_conversations()
            df = loader.to_dataframe(conversations)

            print(f"✅ Chargé {len(conversations)} conversations")
            print(f"✅ DataFrame créé avec {len(df)} lignes")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False

def test_profile_generation():
    """Test de la génération de profils"""
    print("\n🧪 Test 2: Génération de profils")

    try:
        from src.models.profile_generator import ProfileGenerator

        # Créer des données de test
        test_df = pd.DataFrame({
            'session_id': ['test_session'] * 4,
            'user_input': [
                'Bonjour, comment ça marche?',
                'Je ne comprends pas cette partie',
                'Pouvez-vous donner un exemple visuel?',
                'Parfait, je comprends mieux maintenant!'
            ],
            'bot_response': ['Réponse du bot'] * 4,
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='H')
        })

        # Générer le profil
        generator = ProfileGenerator()
        profile = generator.generate_profile(test_df)

        print(f"✅ Profil généré pour {profile.learner_id}")
        print(f"✅ Style d'apprentissage: V:{profile.learning_style.visual:.2f}, A:{profile.learning_style.auditory:.2f}")
        print(f"✅ Confiance: {profile.confidence_level:.2%}")
        print(f"✅ {len(profile.recommended_strategies)} recommandations")

        return True
    except Exception as e:
        print(f"❌ Erreur génération profil: {e}")
        return False

def test_behavioral_analysis():
    """Test de l'analyse comportementale"""
    print("\n🧪 Test 3: Analyse comportementale")

    try:
        from src.features.behavioral_analyzer import BehavioralAnalyzer

        # Créer des données de test
        test_df = pd.DataFrame({
            'session_id': ['test_session'] * 3,
            'user_input': [
                'Euh... je ne sais pas trop',
                'Comment ça marche exactement?',
                'Ah super, merci beaucoup!'
            ],
            'bot_response': ['Réponse'] * 3,
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='H')
        })

        analyzer = BehavioralAnalyzer()
        features = analyzer.analyze_session_behavior(test_df)

        print(f"✅ Analyse comportementale pour {features.session_id}")
        print(f"✅ Taux d'hésitation: {features.hesitation_rate:.2%}")
        print(f"✅ Score d'engagement: {features.engagement_score:.2f}")

        return True
    except Exception as e:
        print(f"❌ Erreur analyse comportementale: {e}")
        return False

def test_integration_bridge():
    """Test du bridge d'intégration"""
    print("\n🧪 Test 4: Bridge d'intégration")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Créer des données de test
            test_data = create_test_data()
            with open(temp_path / "test_session.json", 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)

            # Test du bridge
            from integration_bridge import LegacyAnalysisIntegrator
            integrator = LegacyAnalysisIntegrator()

            output_path = temp_path / "output"
            results = integrator.run_legacy_analysis(str(temp_path), str(output_path))

            if results is not None and not results.empty:
                print(f"✅ Bridge fonctionnel, {len(results)} sessions analysées")
                return True
            else:
                print("❌ Bridge n'a pas retourné de résultats")
                return False

    except Exception as e:
        print(f"❌ Erreur bridge: {e}")
        return False

def main():
    """Exécuter tous les tests"""
    print("🚀 Test du système de profilage des apprenants\n")

    tests = [
        test_data_loading,
        test_profile_generation,
        test_behavioral_analysis,
        test_integration_bridge
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print(f"\n📊 Résultats: {sum(results)}/{len(results)} tests passés")

    if all(results):
        print("🎉 Tous les tests sont passés! Le système est fonctionnel.")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les dépendances et la configuration.")

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
