import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta

class LearnerDashboard:
    """Interactive dashboard for visualizing learner profiles and analytics"""

    def __init__(self, profile_generator, session_data: pd.DataFrame):
        self.profile_generator = profile_generator
        self.session_data = session_data
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Tableau de Bord d'Analyse des Apprenants", 
                   style={'textAlign': 'center', 'marginBottom': 30}),

            # Control panel
            html.Div([
                html.Div([
                    html.Label("Sélectionner un apprenant:"),
                    dcc.Dropdown(
                        id='learner-dropdown',
                        options=[],
                        value=None,
                        placeholder="Choisir un apprenant..."
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label("Période d'analyse:"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        display_format='DD/MM/YYYY'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),

                html.Div([
                    html.Button('Actualiser', id='refresh-button', 
                               style={'backgroundColor': '#007bff', 'color': 'white', 
                                     'border': 'none', 'padding': '10px 20px', 'marginTop': '25px'})
                ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '5%'})
            ], style={'marginBottom': 30}),

            # Main content area
            html.Div([
                # Left column - Profile overview
                html.Div([
                    html.H3("Profil de l'Apprenant"),
                    html.Div(id='profile-overview'),

                    html.H4("Style d'Apprentissage (VARK)", style={'marginTop': 30}),
                    dcc.Graph(id='learning-style-radar'),

                    html.H4("Profil de Personnalité", style={'marginTop': 30}),
                    dcc.Graph(id='personality-radar')
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right column - Analytics
                html.Div([
                    html.H3("Analyse Comportementale"),
                    dcc.Graph(id='behavioral-metrics'),

                    html.H4("Évolution de l'Engagement", style={'marginTop': 30}),
                    dcc.Graph(id='engagement-timeline'),

                    html.H4("Patterns d'Interaction", style={'marginTop': 30}),
                    dcc.Graph(id='interaction-patterns')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'})
            ]),

            # Bottom section - Detailed analytics
            html.Div([
                html.H3("Analyse Détaillée des Sessions"),
                html.Div([
                    html.Div([
                        dcc.Graph(id='sentiment-analysis')
                    ], style={'width': '33%', 'display': 'inline-block'}),

                    html.Div([
                        dcc.Graph(id='comprehension-trends')
                    ], style={'width': '33%', 'display': 'inline-block'}),

                    html.Div([
                        dcc.Graph(id='difficulty-areas')
                    ], style={'width': '33%', 'display': 'inline-block'})
                ])
            ], style={'marginTop': 40}),

            # Recommendations section
            html.Div([
                html.H3("Recommandations Personnalisées"),
                html.Div(id='recommendations-panel')
            ], style={'marginTop': 40, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ])

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output('learner-dropdown', 'options'),
            Input('refresh-button', 'n_clicks')
        )
        def update_learner_options(n_clicks):
            """Update available learners in dropdown"""
            learners = self.session_data['session_id'].unique()
            return [{'label': learner_id, 'value': learner_id} for learner_id in learners]

        @self.app.callback(
            [Output('profile-overview', 'children'),
             Output('learning-style-radar', 'figure'),
             Output('personality-radar', 'figure'),
             Output('behavioral-metrics', 'figure'),
             Output('engagement-timeline', 'figure'),
             Output('interaction-patterns', 'figure'),
             Output('sentiment-analysis', 'figure'),
             Output('comprehension-trends', 'figure'),
             Output('difficulty-areas', 'figure'),
             Output('recommendations-panel', 'children')],
            [Input('learner-dropdown', 'value'),
             Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date')]
        )
        def update_dashboard(selected_learner, start_date, end_date):
            """Update all dashboard components"""

            if not selected_learner:
                empty_fig = go.Figure()
                return ("Sélectionnez un apprenant", empty_fig, empty_fig, empty_fig, 
                       empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "")

            # Filter data for selected learner and date range
            learner_data = self.session_data[self.session_data['session_id'] == selected_learner]

            if start_date and end_date:
                learner_data = learner_data[
                    (pd.to_datetime(learner_data['timestamp']) >= start_date) &
                    (pd.to_datetime(learner_data['timestamp']) <= end_date)
                ]

            # Generate or get profile
            if selected_learner in self.profile_generator.profile_cache:
                profile = self.profile_generator.profile_cache[selected_learner]
            else:
                profile = self.profile_generator.generate_profile(learner_data)

            # Generate all visualizations
            profile_overview = self.create_profile_overview(profile)
            learning_style_fig = self.create_learning_style_radar(profile)
            personality_fig = self.create_personality_radar(profile)
            behavioral_fig = self.create_behavioral_metrics(profile)
            engagement_fig = self.create_engagement_timeline(learner_data)
            interaction_fig = self.create_interaction_patterns(learner_data)
            sentiment_fig = self.create_sentiment_analysis(learner_data)
            comprehension_fig = self.create_comprehension_trends(learner_data)
            difficulty_fig = self.create_difficulty_areas(profile)
            recommendations = self.create_recommendations_panel(profile)

            return (profile_overview, learning_style_fig, personality_fig, behavioral_fig,
                   engagement_fig, interaction_fig, sentiment_fig, comprehension_fig,
                   difficulty_fig, recommendations)

    def create_profile_overview(self, profile) -> html.Div:
        """Create profile overview component"""
        return html.Div([
            html.Div([
                html.H5("ID Apprenant"),
                html.P(profile.learner_id)
            ], style={'marginBottom': 15}),

            html.Div([
                html.H5("Niveau de Confiance du Profil"),
                html.Div([
                    html.Div(style={
                        'width': f'{profile.confidence_level * 100}%',
                        'height': '20px',
                        'backgroundColor': '#28a745' if profile.confidence_level > 0.7 else '#ffc107' if profile.confidence_level > 0.4 else '#dc3545',
                        'borderRadius': '10px'
                    })
                ], style={'width': '100%', 'height': '20px', 'backgroundColor': '#e9ecef', 'borderRadius': '10px'}),
                html.P(f"{profile.confidence_level:.1%}", style={'textAlign': 'center', 'marginTop': '5px'})
            ], style={'marginBottom': 15}),

            html.Div([
                html.H5("Dernière Mise à Jour"),
                html.P(profile.last_updated.strftime("%d/%m/%Y %H:%M"))
            ])
        ])

    def create_learning_style_radar(self, profile) -> go.Figure:
        """Create VARK learning style radar chart"""
        categories = ['Visuel', 'Auditif', 'Lecture/Écriture', 'Kinesthésique']
        values = [
            profile.learning_style.visual,
            profile.learning_style.auditory,
            profile.learning_style.reading,
            profile.learning_style.kinesthetic
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Style d\'Apprentissage',
            line_color='rgb(50, 171, 96)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=300
        )

        return fig

    def create_personality_radar(self, profile) -> go.Figure:
        """Create personality radar chart"""
        categories = ['Extraversion', 'Intuition', 'Pensée', 'Jugement']
        values = [
            profile.personality.extraversion,
            profile.personality.intuition,
            profile.personality.thinking,
            profile.personality.judging
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Personnalité',
            line_color='rgb(255, 99, 71)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=300
        )

        return fig

    def create_behavioral_metrics(self, profile) -> go.Figure:
        """Create behavioral metrics bar chart"""
        metrics = ['Engagement', 'Persévérance', 'Recherche d\'Aide', 
                  'Collaboration', 'Autorégulation']
        values = [
            profile.behavioral.engagement_level,
            profile.behavioral.persistence,
            profile.behavioral.help_seeking,
            profile.behavioral.collaboration_preference,
            profile.behavioral.self_regulation
        ]

        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, 
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ])

        fig.update_layout(
            title="Métriques Comportementales",
            yaxis_title="Score (0-1)",
            height=300
        )

        return fig

    def create_engagement_timeline(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create engagement timeline"""
        if learner_data.empty or 'timestamp' not in learner_data.columns:
            return go.Figure()

        # Calculate engagement over time (simplified)
        learner_data['timestamp'] = pd.to_datetime(learner_data['timestamp'])
        learner_data = learner_data.sort_values('timestamp')
        learner_data['message_length'] = learner_data['user_input'].str.len()

        # Rolling average for smoothing
        learner_data['engagement_proxy'] = learner_data['message_length'].rolling(window=3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=learner_data['timestamp'],
            y=learner_data['engagement_proxy'],
            mode='lines+markers',
            name='Engagement',
            line=dict(color='#17a2b8')
        ))

        fig.update_layout(
            title="Évolution de l'Engagement",
            xaxis_title="Temps",
            yaxis_title="Niveau d'Engagement (proxy)",
            height=300
        )

        return fig

    def create_interaction_patterns(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create interaction patterns visualization"""
        if learner_data.empty:
            return go.Figure()

        # Calculate interaction statistics
        total_messages = len(learner_data)
        questions = learner_data['user_input'].str.contains(r'\?', na=False).sum()
        short_messages = (learner_data['user_input'].str.len() < 20).sum()
        long_messages = (learner_data['user_input'].str.len() > 100).sum()

        categories = ['Messages Courts', 'Messages Longs', 'Questions', 'Déclarations']
        values = [short_messages, long_messages, questions, total_messages - questions]

        fig = go.Figure(data=[go.Pie(labels=categories, values=values, hole=.3)])
        fig.update_layout(
            title="Patterns d'Interaction",
            height=300
        )

        return fig

    def create_sentiment_analysis(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create sentiment analysis visualization"""
        if learner_data.empty or 'sentiment_score' not in learner_data.columns:
            return go.Figure()

        # Create sentiment distribution
        sentiment_counts = learner_data['sentiment_score'].value_counts().sort_index()

        fig = go.Figure(data=[
            go.Bar(x=[f"{int(score)} étoiles" for score in sentiment_counts.index], 
                   y=sentiment_counts.values,
                   marker_color='lightblue')
        ])

        fig.update_layout(
            title="Distribution des Sentiments",
            xaxis_title="Score de Sentiment",
            yaxis_title="Fréquence",
            height=300
        )

        return fig

    def create_comprehension_trends(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create comprehension trends visualization"""
        if learner_data.empty or 'comprehension_score' not in learner_data.columns:
            return go.Figure()

        # Group by time periods and calculate average comprehension
        learner_data['timestamp'] = pd.to_datetime(learner_data['timestamp'])
        learner_data['date'] = learner_data['timestamp'].dt.date

        daily_comprehension = learner_data.groupby('date')['comprehension_score'].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_comprehension.index,
            y=daily_comprehension.values,
            mode='lines+markers',
            name='Compréhension',
            line=dict(color='#28a745')
        ))
"""
Simplified dashboard functionality
"""
import pandas as pd
import json
from pathlib import Path

class LearnerDashboard:
    """Simplified dashboard for basic analytics"""

    def __init__(self, profile_generator, session_data: pd.DataFrame):
        self.profile_generator = profile_generator
        self.session_data = session_data

    def generate_summary_report(self, output_path: str = "dashboard_report.html"):
        """Generate a simple HTML report"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learner Analytics Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .positive { color: green; }
                .negative { color: red; }
                .neutral { color: orange; }
            </style>
        </head>
        <body>
            <h1>📊 Rapport d'Analyse des Apprenants</h1>
        """

        # Basic statistics
        total_sessions = self.session_data['session_id'].nunique()
        total_messages = len(self.session_data)
        avg_messages_per_session = total_messages / max(total_sessions, 1)

        html_content += f"""
            <div class="metric">
                <h3>Statistiques Générales</h3>
                <p><strong>Sessions totales:</strong> {total_sessions}</p>
                <p><strong>Messages totaux:</strong> {total_messages}</p>
                <p><strong>Messages par session:</strong> {avg_messages_per_session:.1f}</p>
            </div>
        """

        # Session analysis
        session_stats = []
        for session_id in self.session_data['session_id'].unique():
            session_messages = self.session_data[self.session_data['session_id'] == session_id]
            session_stats.append({
                'session_id': session_id,
                'message_count': len(session_messages),
                'avg_message_length': session_messages['user_input'].str.len().mean()
            })

        html_content += "<div class='metric'><h3>Analyse par Session</h3><ul>"
        for stat in session_stats[:10]:  # Show first 10 sessions
            html_content += f"<li><strong>{stat['session_id']}:</strong> {stat['message_count']} messages, longueur moyenne: {stat['avg_message_length']:.0f} caractères</li>"
        html_content += "</ul></div>"

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Rapport généré: {output_path}")

    def run(self, debug=False, port=8050):
        """Simplified run method"""
        print(f"📊 Dashboard simplifié - Génération du rapport HTML...")
        self.generate_summary_report()
        print("💡 Pour un dashboard interactif complet, installez: pip install dash plotly")
        fig.update_layout(
            title="Évolution de la Compréhension",
            xaxis_title="Date",
            yaxis_title="Score de Compréhension",
            height=300
        )

        return fig

    def create_difficulty_areas(self, profile) -> go.Figure:
        """Create difficulty areas visualization"""
        if not profile.difficulty_areas and not profile.strengths:
            return go.Figure()

        # Combine difficulties and strengths
        categories = profile.difficulty_areas + profile.strengths
        values = [-1] * len(profile.difficulty_areas) + [1] * len(profile.strengths)
        colors = ['red'] * len(profile.difficulty_areas) + ['green'] * len(profile.strengths)

        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=colors)
        ])

        fig.update_layout(
            title="Forces et Difficultés",
            xaxis_title="Domaines",
            yaxis_title="Impact",
            height=300
        )

        return fig

    def create_recommendations_panel(self, profile) -> html.Div:
        """Create recommendations panel"""
        if not profile.recommended_strategies:
            return html.Div("Aucune recommandation disponible")

        recommendation_items = []
        for i, rec in enumerate(profile.recommended_strategies, 1):
            recommendation_items.append(
                html.Div([
                    html.Strong(f"{i}. "),
                    html.Span(rec)
                ], style={'marginBottom': '10px'})
            )

        return html.Div([
            html.H4("Stratégies Recommandées"),
            html.Div(recommendation_items)
        ])

    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)

# Usage example function
def create_sample_dashboard():
    """Create a sample dashboard with mock data"""
    from src.models.profile_generator import ProfileGenerator

    # Create sample data
    sample_data = pd.DataFrame({
        'session_id': ['learner_001'] * 10,
        'user_input': [
            'Bonjour, comment ça marche?',
            'Je ne comprends pas cette partie',
            'Pouvez-vous expliquer davantage?',
            'C\'est intéressant, merci!',
            'J\'ai encore des questions...',
            'Comment faire cela étape par étape?',
            'Parfait, je comprends mieux maintenant',
            'Y a-t-il d\'autres exemples?',
            'C\'est plus clair, merci beaucoup',
            'Je pense avoir saisi le concept'
        ],
        'bot_response': ['Réponse du bot'] * 10,
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
        'sentiment_score': [3, 2, 2, 4, 3, 3, 5, 4, 5, 4],
        'comprehension_score': [0.3, 0.2, 0.4, 0.6, 0.5, 0.7, 0.8, 0.7, 0.9, 0.8]
    })

    profile_gen = ProfileGenerator()
    dashboard = LearnerDashboard(profile_gen, sample_data)

    return dashboard

if __name__ == "__main__":
    dashboard = create_sample_dashboard()
    dashboard.run(debug=True)
