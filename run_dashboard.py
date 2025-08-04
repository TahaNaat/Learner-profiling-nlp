#!/usr/bin/env python3
"""
Script pour lancer le dashboard interactif avec les vraies données
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.profile_generator import ProfileGenerator
from src.visualization.dashboard import LearnerDashboard
from src.data.loaders import ConversationLoader
from src.config.settings import config

def main():
    print("🚀 Lancement du Dashboard Interactif...")
    
    # Charger les données de conversation
    print("📊 Chargement des données...")
    conversation_loader = ConversationLoader(config.data_dir)
    conversations = conversation_loader.load_conversations()
    
    if not conversations:
        print("❌ Aucune conversation trouvée!")
        return
    
    # Convertir en DataFrame
    df_conversations = conversation_loader.to_dataframe(conversations)
    print(f"✅ {len(conversations)} conversations chargées")
    
    # Créer le générateur de profils
    profile_generator = ProfileGenerator()
    
    # Créer et lancer le dashboard
    print("🎯 Création du dashboard...")
    dashboard = LearnerDashboard(profile_generator, df_conversations)
    
    print("🌐 Lancement du serveur web...")
    print("📱 Ouvrez votre navigateur sur: http://localhost:8050")
    print("⏹️  Appuyez sur Ctrl+C pour arrêter")
    
    try:
        dashboard.run(debug=True, port=8050)
    except KeyboardInterrupt:
        print("\n👋 Dashboard arrêté")

if __name__ == "__main__":
    main() 