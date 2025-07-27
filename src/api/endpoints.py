from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import json
import uvicorn

from src.models.profile_generator import ProfileGenerator, LearnerProfile
from src.features.behavioral_analyzer import BehavioralAnalyzer
from src.data.loaders import ConversationLoader, ConversationMessage

# Pydantic models for API
class ConversationInput(BaseModel):
    user_input: str
    bot_response: str
    timestamp: datetime
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

class BatchConversationInput(BaseModel):
    conversations: List[ConversationInput]

class ProfileResponse(BaseModel):
    learner_id: str
    learning_style: Dict[str, float]
    personality: Dict[str, float]
    cognitive: Dict[str, float]
    behavioral: Dict[str, float]
    difficulty_areas: List[str]
    strengths: List[str]
    recommended_strategies: List[str]
    confidence_level: float
    last_updated: datetime

class AnalysisRequest(BaseModel):
    session_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class RecommendationResponse(BaseModel):
    learner_id: str
    recommendations: List[str]
    priority_areas: List[str]
    confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="Learner Profiling API",
    description="API pour l'analyse et le profilage des apprenants en temps réel",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
profile_generator = ProfileGenerator()
behavioral_analyzer = BehavioralAnalyzer()

# In-memory storage for conversations (in production, use a database)
conversation_storage: Dict[str, List[ConversationMessage]] = {}

@app.post("/api/conversations/single")
async def add_single_conversation(conversation: ConversationInput):
    """Add a single conversation exchange"""
    try:
        conv_message = ConversationMessage(
            user_input=conversation.user_input,
            bot_response=conversation.bot_response,
            timestamp=conversation.timestamp,
            session_id=conversation.session_id,
            metadata=conversation.metadata or {}
        )

        if conversation.session_id not in conversation_storage:
            conversation_storage[conversation.session_id] = []

        conversation_storage[conversation.session_id].append(conv_message)

        return {
            "status": "success",
            "message": "Conversation ajoutée avec succès",
            "session_id": conversation.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'ajout: {str(e)}")

@app.post("/api/conversations/batch")
async def add_batch_conversations(batch: BatchConversationInput, background_tasks: BackgroundTasks):
    """Add multiple conversations and trigger profile update"""
    try:
        for conv_input in batch.conversations:
            conv_message = ConversationMessage(
                user_input=conv_input.user_input,
                bot_response=conv_input.bot_response,
                timestamp=conv_input.timestamp,
                session_id=conv_input.session_id,
                metadata=conv_input.metadata or {}
            )

            if conv_input.session_id not in conversation_storage:
                conversation_storage[conv_input.session_id] = []

            conversation_storage[conv_input.session_id].append(conv_message)

        # Schedule background profile update
        session_ids = list(set([conv.session_id for conv in batch.conversations]))
        background_tasks.add_task(update_profiles_background, session_ids)

        return {
            "status": "success",
            "message": f"{len(batch.conversations)} conversations ajoutées",
            "sessions_affected": session_ids
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'ajout en lot: {str(e)}")

@app.get("/api/profile/{learner_id}")
async def get_learner_profile(learner_id: str) -> ProfileResponse:
    """Get comprehensive learner profile"""
    try:
        if learner_id not in conversation_storage:
            raise HTTPException(status_code=404, detail="Apprenant non trouvé")

        # Convert conversations to DataFrame
        conversations = conversation_storage[learner_id]
        data = []
        for conv in conversations:
            data.append({
                'user_input': conv.user_input,
                'bot_response': conv.bot_response,
                'timestamp': conv.timestamp,
                'session_id': conv.session_id,
                'metadata': conv.metadata
            })

        df = pd.DataFrame(data)

        # Generate or update profile
        if learner_id in profile_generator.profile_cache:
            profile = profile_generator.update_profile(learner_id, df)
        else:
            profile = profile_generator.generate_profile(df)

        # Convert to response format
        response = ProfileResponse(
            learner_id=profile.learner_id,
            learning_style={
                'visual': profile.learning_style.visual,
                'auditory': profile.learning_style.auditory,
                'reading': profile.learning_style.reading,
                'kinesthetic': profile.learning_style.kinesthetic
            },
            personality={
                'extraversion': profile.personality.extraversion,
                'intuition': profile.personality.intuition,
                'thinking': profile.personality.thinking,
                'judging': profile.personality.judging
            },
            cognitive={
                'processing_speed': profile.cognitive.processing_speed,
                'working_memory': profile.cognitive.working_memory,
                'analytical_thinking': profile.cognitive.analytical_thinking,
                'creative_thinking': profile.cognitive.creative_thinking,
                'attention_span': profile.cognitive.attention_span
            },
            behavioral={
                'engagement_level': profile.behavioral.engagement_level,
                'persistence': profile.behavioral.persistence,
                'help_seeking': profile.behavioral.help_seeking,
                'collaboration_preference': profile.behavioral.collaboration_preference,
                'self_regulation': profile.behavioral.self_regulation
            },
            difficulty_areas=profile.difficulty_areas,
            strengths=profile.strengths,
            recommended_strategies=profile.recommended_strategies,
            confidence_level=profile.confidence_level,
            last_updated=profile.last_updated
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/api/recommendations/{learner_id}")
async def get_recommendations(learner_id: str) -> RecommendationResponse:
    """Get personalized learning recommendations"""
    try:
        if learner_id not in profile_generator.profile_cache:
            raise HTTPException(status_code=404, detail="Profil d'apprenant non trouvé")

        profile = profile_generator.profile_cache[learner_id]

        # Identify priority areas based on low scores
        priority_areas = []

        # Check cognitive areas
        if profile.cognitive.processing_speed < 0.4:
            priority_areas.append("Vitesse de traitement")
        if profile.cognitive.working_memory < 0.4:
            priority_areas.append("Mémoire de travail")
        if profile.cognitive.attention_span < 0.4:
            priority_areas.append("Attention et concentration")

        # Check behavioral areas
        if profile.behavioral.engagement_level < 0.4:
            priority_areas.append("Engagement et motivation")
        if profile.behavioral.persistence < 0.4:
            priority_areas.append("Persévérance")
        if profile.behavioral.self_regulation < 0.3:
            priority_areas.append("Autorégulation")

        return RecommendationResponse(
            learner_id=learner_id,
            recommendations=profile.recommended_strategies,
            priority_areas=priority_areas,
            confidence=profile.confidence_level
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des recommandations: {str(e)}")

@app.get("/api/analytics/behavioral/{learner_id}")
async def get_behavioral_analysis(learner_id: str):
    """Get detailed behavioral analysis for a learner"""
    try:
        if learner_id not in conversation_storage:
            raise HTTPException(status_code=404, detail="Données d'apprenant non trouvées")

        # Convert conversations to DataFrame
        conversations = conversation_storage[learner_id]
        data = []
        for conv in conversations:
            data.append({
                'user_input': conv.user_input,
                'bot_response': conv.bot_response,
                'timestamp': conv.timestamp,
                'session_id': conv.session_id
            })

        df = pd.DataFrame(data)
"""
Simplified API endpoints using Flask instead of FastAPI
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Simplified API without FastAPI dependency
class SimpleAPI:
    """Simplified API for basic functionality"""

    def __init__(self, profile_generator):
        self.profile_generator = profile_generator
        self.conversation_storage: Dict[str, List[Dict]] = {}

    def add_conversation(self, session_id: str, user_input: str, bot_response: str, timestamp: str = None):
        """Add a conversation exchange"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if session_id not in self.conversation_storage:
            self.conversation_storage[session_id] = []

        self.conversation_storage[session_id].append({
            'user_input': user_input,
            'bot_response': bot_response,
            'timestamp': timestamp
        })

        return {"status": "success", "session_id": session_id}

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        """Get learner profile"""
        if session_id not in self.conversation_storage:
            return {"error": "Session not found"}

        # Convert to DataFrame
        data = []
        for conv in self.conversation_storage[session_id]:
            conv['session_id'] = session_id
            data.append(conv)

        df = pd.DataFrame(data)
        profile = self.profile_generator.generate_profile(df)

        return {
            "learner_id": profile.learner_id,
            "learning_style": {
                "visual": profile.learning_style.visual,
                "auditory": profile.learning_style.auditory,
                "reading": profile.learning_style.reading,
                "kinesthetic": profile.learning_style.kinesthetic
            },
            "confidence_level": profile.confidence_level,
            "strengths": profile.strengths,
            "difficulties": profile.difficulty_areas,
            "recommendations": profile.recommended_strategies
        }

    def get_sessions(self) -> List[str]:
        """Get list of all sessions"""
        return list(self.conversation_storage.keys())

    def export_data(self, output_path: str = "api_data.json"):
        """Export all data"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_storage, f, indent=2, ensure_ascii=False)
        return {"status": "exported", "path": output_path}

def create_simple_api(profile_generator):
    """Factory function to create simple API"""
    return SimpleAPI(profile_generator)

# For compatibility
def create_app():
    """Create a simple Flask-like app structure"""
    print("💡 Pour une API REST complète, installez: pip install fastapi uvicorn")
    print("📝 Utilisation de l'API simplifiée à la place")
    return None
        # Perform behavioral analysis
        behavioral_features = behavioral_analyzer.analyze_session_behavior(df)

        return {
            "learner_id": learner_id,
            "behavioral_analysis": {
                "hesitation_rate": behavioral_features.hesitation_rate,
                "question_complexity": behavioral_features.question_complexity,
                "engagement_score": behavioral_features.engagement_score,
                "interaction_patterns": behavioral_features.interaction_patterns,
                "semantic_clusters": behavioral_features.semantic_clusters,
                "learning_pace": behavioral_features.learning_pace,
                "confusion_indicators": behavioral_features.confusion_indicators
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse comportementale: {str(e)}")

@app.get("/api/similar-learners/{learner_id}")
async def get_similar_learners(learner_id: str, limit: int = 5):
    """Find similar learners based on profile characteristics"""
    try:
        similar_learners = profile_generator.get_similar_learners(learner_id, limit)

        if not similar_learners:
            return {
                "learner_id": learner_id,
                "similar_learners": [],
                "message": "Aucun apprenant similaire trouvé"
            }

        return {
            "learner_id": learner_id,
            "similar_learners": similar_learners,
            "count": len(similar_learners)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche d'apprenants similaires: {str(e)}")

@app.get("/api/sessions/{learner_id}")
async def get_session_summary(learner_id: str):
    """Get session summary and statistics"""
    try:
        if learner_id not in conversation_storage:
            raise HTTPException(status_code=404, detail="Sessions non trouvées")

        conversations = conversation_storage[learner_id]

        # Calculate session statistics
        total_messages = len(conversations)
        questions = sum(1 for conv in conversations if '?' in conv.user_input)
        avg_message_length = sum(len(conv.user_input) for conv in conversations) / max(total_messages, 1)

        # Time span analysis
        if conversations:
            start_time = min(conv.timestamp for conv in conversations)
            end_time = max(conv.timestamp for conv in conversations)
            duration = (end_time - start_time).total_seconds() / 3600  # hours
        else:
            start_time = end_time = None
            duration = 0

        return {
            "learner_id": learner_id,
            "session_stats": {
                "total_messages": total_messages,
                "questions_asked": questions,
                "avg_message_length": round(avg_message_length, 2),
                "session_duration_hours": round(duration, 2),
                "start_time": start_time,
                "end_time": end_time
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des statistiques: {str(e)}")

@app.delete("/api/conversations/{learner_id}")
async def delete_learner_data(learner_id: str):
    """Delete all data for a specific learner"""
    try:
        if learner_id in conversation_storage:
            del conversation_storage[learner_id]

        if learner_id in profile_generator.profile_cache:
            del profile_generator.profile_cache[learner_id]

        return {
            "status": "success",
            "message": f"Toutes les données de {learner_id} ont été supprimées"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_learners": len(conversation_storage),
        "cached_profiles": len(profile_generator.profile_cache)
    }

# Background tasks
async def update_profiles_background(session_ids: List[str]):
    """Background task to update profiles"""
    for session_id in session_ids:
        if session_id in conversation_storage:
            conversations = conversation_storage[session_id]
            data = []
            for conv in conversations:
                data.append({
                    'user_input': conv.user_input,
                    'bot_response': conv.bot_response,
                    'timestamp': conv.timestamp,
                    'session_id': conv.session_id
                })

            df = pd.DataFrame(data)
            profile_generator.update_profile(session_id, df)

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
