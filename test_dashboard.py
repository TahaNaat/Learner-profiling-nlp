#!/usr/bin/env python3
"""
Script de test pour le dashboard
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.profile_generator import ProfileGenerator
from src.visualization.dashboard import LearnerDashboard

def main():
    print("🧪 Test du Dashboard...")
    
    # Créer des données de test
    test_data = pd.DataFrame({
        'session_id': ['test_learner_001'] * 5,
        'user_input': [
            'Bonjour, comment ça marche?',
            'Je ne comprends pas cette partie',
            'Pouvez-vous expliquer davantage?',
            'C\'est intéressant, merci!',
            'J\'ai encore des questions...'
        ],
        'bot_response': ['Réponse du bot'] * 5,
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
    })
    
    print(f"✅ Données de test créées: {len(test_data)} messages")
    
    # Créer le générateur de profils
    profile_generator = ProfileGenerator()
    print("✅ Générateur de profils créé")
    
    # Créer le dashboard
    try:
        dashboard = LearnerDashboard(profile_generator, test_data)
        print("✅ Dashboard créé avec succès")
        
        # Test de génération de rapport
        dashboard.generate_summary_report("test_report.html")
        print("✅ Rapport de test généré")
        
        print("🚀 Lancement du dashboard...")
        print("📱 Ouvrez http://localhost:8050 dans votre navigateur")
        print("⏹️  Appuyez sur Ctrl+C pour arrêter")
        
        dashboard.run(debug=True, port=8050)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 