#!/usr/bin/env python3
"""
Script pour générer un rapport HTML statique avec les résultats
"""

import sys
import pandas as pd
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.profile_generator import ProfileGenerator
from src.visualization.dashboard import LearnerDashboard
from src.data.loaders import ConversationLoader
from src.config.settings import config

def main():
    print("📊 Génération du rapport HTML...")
    
    # Charger les données de conversation
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
    
    # Créer le dashboard et générer le rapport
    dashboard = LearnerDashboard(profile_generator, df_conversations)
    
    # Générer le rapport HTML
    output_path = "rapport_analyse_apprenants.html"
    dashboard.generate_summary_report(output_path)
    
    print(f"✅ Rapport généré: {output_path}")
    print("🌐 Ouvrez le fichier dans votre navigateur pour voir les résultats")

if __name__ == "__main__":
    main() 