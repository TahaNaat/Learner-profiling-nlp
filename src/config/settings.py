import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for NLP models"""
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    question_threshold: float = 0.7
    similarity_threshold: float = 0.9
    hesitation_patterns: List[str] = None

    def __post_init__(self):
        if self.hesitation_patterns is None:
            self.hesitation_patterns = [
                "euh", "mmh", "hmm", "ben", "alors", "donc", 
                "comment dire", "je sais pas", "peut-être"
            ]

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    batch_size: int = 32
    max_sequence_length: int = 512
    min_session_length: int = 3
    cache_embeddings: bool = True

@dataclass
class ProfileConfig:
    """Configuration for learner profiling"""
    profile_update_frequency: str = "daily"  # daily, weekly, monthly
    personality_dimensions: List[str] = None
    learning_styles: List[str] = None

    def __post_init__(self):
        if self.personality_dimensions is None:
            self.personality_dimensions = ["introversion", "intuition", "thinking", "judging"]
        if self.learning_styles is None:
            self.learning_styles = ["visual", "auditory", "reading", "kinesthetic"]

@dataclass
class AppConfig:
    """Main application configuration - adapted for your project structure"""
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = None
    output_dir: Path = None
    models: ModelConfig = None
    processing: ProcessingConfig = None
    profiling: ProfileConfig = None

    def __post_init__(self):
        if self.data_dir is None:
            # Check multiple possible locations for conversations
            possible_paths = [
                self.project_root / "conversations" / "conversations",  # Your actual structure
                self.project_root / "conversations",
                self.project_root / "data" / "conversations",
                Path.cwd() / "conversations"
            ]

            for path in possible_paths:
                if path.exists():
                    self.data_dir = path
                    break
            else:
                # Default fallback
                self.data_dir = self.project_root / "conversations" / "conversations"

        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.models is None:
            self.models = ModelConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.profiling is None:
            self.profiling = ProfileConfig()

        # Create directories if they don't exist
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "profiles").mkdir(exist_ok=True)
            (self.output_dir / "analytics").mkdir(exist_ok=True)
        except Exception as e:
            print(f"⚠️ Attention: Impossible de créer les dossiers: {e}")

# Global configuration instance
config = AppConfig()
