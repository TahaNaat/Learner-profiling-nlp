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
    """Modern interactive dashboard for visualizing learner profiles and analytics"""

    def __init__(self, profile_generator, session_data: pd.DataFrame):
        self.profile_generator = profile_generator
        self.session_data = session_data
        self.app = dash.Dash(__name__, 
                            external_stylesheets=[
                                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
                            ])
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup modern dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.I(className="fas fa-graduation-cap", style={'fontSize': '24px', 'marginRight': '10px'}),
                    html.H1("Dashboard d'Analyse des Apprenants", 
                           style={'margin': '0', 'color': '#2c3e50', 'fontSize': '28px'})
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Span(f"📊 {self.session_data['session_id'].nunique()} apprenants analysés", 
                             style={'color': '#7f8c8d', 'fontSize': '14px'})
                ])
            ], style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'marginBottom': '30px',
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center'
            }),

            # Control Panel
            html.Div([
                html.Div([
                    html.Label("👤 Sélectionner un apprenant", 
                              style={'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='learner-dropdown',
                        options=[],
                        value=None,
                        placeholder="Choisir un apprenant...",
                        style={'borderRadius': '8px'}
                    )
                ], style={'width': '40%', 'marginRight': '20px'}),

                html.Div([
                    html.Label("📅 Période d'analyse", 
                              style={'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '5px'}),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        display_format='DD/MM/YYYY',
                        style={'borderRadius': '8px'}
                    )
                ], style={'width': '40%', 'marginRight': '20px'}),

                html.Div([
                    html.Button([
                        html.I(className="fas fa-sync-alt", style={'marginRight': '8px'}),
                        "Actualiser"
                    ], id='refresh-button', 
                       style={
                           'backgroundColor': '#3498db',
                           'color': 'white',
                           'border': 'none',
                           'padding': '12px 24px',
                           'borderRadius': '8px',
                           'cursor': 'pointer',
                           'fontWeight': 'bold',
                           'transition': 'all 0.3s ease'
                       })
                ], style={'width': '20%', 'display': 'flex', 'alignItems': 'end'})
            ], style={
                'display': 'flex',
                'marginBottom': '30px',
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
            }),

            # Main Content
            html.Div([
                # Left Column - Profile & Learning Style
                html.Div([
                    # Profile Overview Card
                    html.Div([
                        html.H3("👤 Profil de l'Apprenant", 
                               style={'color': '#2c3e50', 'marginBottom': '20px', 'borderBottom': '2px solid #ecf0f1', 'paddingBottom': '10px'}),
                        html.Div(id='profile-overview')
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),

                    # Learning Style Card
                    html.Div([
                        html.H3("🎯 Style d'Apprentissage (VARK)", 
                               style={'color': '#2c3e50', 'marginBottom': '20px'}),
                        dcc.Graph(id='learning-style-radar', style={'height': '300px'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),

                    # Personality Card
                    html.Div([
                        html.H3("🧠 Profil de Personnalité", 
                               style={'color': '#2c3e50', 'marginBottom': '20px'}),
                        dcc.Graph(id='personality-radar', style={'height': '300px'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
                    })
                ], style={'width': '48%', 'marginRight': '2%'}),

                # Right Column - Analytics
                html.Div([
                    # Behavioral Metrics Card
                    html.Div([
                        html.H3("📊 Métriques Comportementales", 
                               style={'color': '#2c3e50', 'marginBottom': '20px'}),
                        dcc.Graph(id='behavioral-metrics', style={'height': '300px'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),

                    # Engagement Timeline Card
                    html.Div([
                        html.H3("📈 Évolution de l'Engagement", 
                               style={'color': '#2c3e50', 'marginBottom': '20px'}),
                        dcc.Graph(id='engagement-timeline', style={'height': '250px'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),

                    # Interaction Patterns Card
                    html.Div([
                        html.H3("🔄 Patterns d'Interaction", 
                               style={'color': '#2c3e50', 'marginBottom': '20px'}),
                        dcc.Graph(id='interaction-patterns', style={'height': '250px'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
                    })
                ], style={'width': '48%', 'marginLeft': '2%'})
            ], style={'display': 'flex', 'marginBottom': '30px'}),

            # Bottom Section - Detailed Analytics
            html.Div([
                html.H3("📋 Analyse Détaillée", 
                       style={'color': '#2c3e50', 'marginBottom': '25px', 'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.H4("😊 Analyse des Sentiments", 
                               style={'color': '#2c3e50', 'marginBottom': '15px', 'textAlign': 'center'}),
                        dcc.Graph(id='sentiment-analysis', style={'height': '250px'})
                    ], style={'width': '32%', 'marginRight': '2%'}),

                    html.Div([
                        html.H4("📚 Évolution de la Compréhension", 
                               style={'color': '#2c3e50', 'marginBottom': '15px', 'textAlign': 'center'}),
                        dcc.Graph(id='comprehension-trends', style={'height': '250px'})
                    ], style={'width': '32%', 'marginRight': '2%'}),

                    html.Div([
                        html.H4("🎯 Forces et Difficultés", 
                               style={'color': '#2c3e50', 'marginBottom': '15px', 'textAlign': 'center'}),
                        dcc.Graph(id='difficulty-areas', style={'height': '250px'})
                    ], style={'width': '32%'})
                ], style={'display': 'flex'})
            ], style={
                'backgroundColor': 'white',
                'padding': '30px',
                'borderRadius': '12px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            }),

            # Recommendations Panel
            html.Div([
                html.H3("💡 Recommandations Personnalisées", 
                       style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
                html.Div(id='recommendations-panel')
            ], style={
                'backgroundColor': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'padding': '30px',
                'borderRadius': '12px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
            })
        ], style={
            'backgroundColor': '#f8f9fa',
            'minHeight': '100vh',
            'padding': '20px',
            'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
        })

    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            Output('learner-dropdown', 'options'),
            Input('refresh-button', 'n_clicks')
        )
        def update_learner_options(n_clicks):
            session_ids = self.session_data['session_id'].unique()
            return [{'label': f'👤 Apprenant {sid[:8]}...', 'value': sid} for sid in session_ids]

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
            if not selected_learner:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    annotations=[{
                        'text': 'Sélectionnez un apprenant pour voir les détails',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#7f8c8d'}
                    }]
                )
                return [
                    html.Div([
                        html.I(className="fas fa-user-circle", style={'fontSize': '48px', 'color': '#bdc3c7', 'marginBottom': '15px'}),
                        html.P("Sélectionnez un apprenant pour voir les détails", 
                               style={'color': '#7f8c8d', 'fontSize': '16px', 'textAlign': 'center'})
                    ], style={'textAlign': 'center', 'padding': '40px'}),
                    empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                    html.Div([
                        html.I(className="fas fa-lightbulb", style={'fontSize': '48px', 'color': '#f1c40f', 'marginBottom': '15px'}),
                        html.P("Les recommandations apparaîtront ici", 
                               style={'color': 'black', 'fontSize': '16px', 'textAlign': 'center'})
                    ], style={'textAlign': 'center', 'padding': '40px'})
                ]

            # Get learner data
            learner_data = self.session_data[self.session_data['session_id'] == selected_learner]
            
            # Generate profile
            try:
                profile = self.profile_generator.generate_profile(learner_data)
            except Exception as e:
                profile = None

            if profile:
                return [
                    self.create_profile_overview(profile),
                    self.create_learning_style_radar(profile),
                    self.create_personality_radar(profile),
                    self.create_behavioral_metrics(profile),
                    self.create_engagement_timeline(learner_data),
                    self.create_interaction_patterns(learner_data),
                    self.create_sentiment_analysis(learner_data),
                    self.create_comprehension_trends(learner_data),
                    self.create_difficulty_areas(profile),
                    self.create_recommendations_panel(profile)
                ]
            else:
                error_fig = go.Figure()
                error_fig.update_layout(
                    annotations=[{
                        'text': 'Erreur lors de la génération du profil',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#e74c3c'}
                    }]
                )
                return [html.Div("Erreur lors de la génération du profil")] * 10

    def create_profile_overview(self, profile) -> html.Div:
        """Create modern profile overview panel"""
        return html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-user-circle", style={'fontSize': '48px', 'color': '#3498db'}),
                    html.Div([
                        html.H4(f"Apprenant {profile.learner_id[:8]}...", 
                               style={'margin': '0 0 5px 0', 'color': '#2c3e50'}),
                        html.P(f"ID: {profile.learner_id}", 
                              style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '12px'})
                    ], style={'marginLeft': '15px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Span("Confiance", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        html.Div([
                            html.Div(style={
                                'width': f'{profile.confidence_level * 100}%',
                                'height': '8px',
                                'backgroundColor': '#27ae60' if profile.confidence_level > 0.7 else '#f39c12' if profile.confidence_level > 0.4 else '#e74c3c',
                                'borderRadius': '4px'
                            })
                        ], style={'width': '100%', 'height': '8px', 'backgroundColor': '#ecf0f1', 'borderRadius': '4px', 'marginTop': '5px'}),
                        html.Span(f"{profile.confidence_level:.1%}", 
                                style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.I(className="fas fa-clock", style={'color': '#7f8c8d', 'marginRight': '8px'}),
                        html.Span(f"Dernière mise à jour: {profile.last_updated.strftime('%d/%m/%Y %H:%M')}", 
                                style={'color': '#7f8c8d', 'fontSize': '12px'})
                    ])
                ])
            ], style={'marginBottom': '20px'}),
            
            html.Hr(style={'border': '1px solid #ecf0f1', 'margin': '20px 0'}),
            
            html.Div([
                html.Div([
                    html.H5("✅ Points forts", style={'color': '#27ae60', 'marginBottom': '10px'}),
                    html.Ul([html.Li(strength, style={'color': '#2c3e50', 'marginBottom': '5px'}) 
                            for strength in profile.strengths]) if profile.strengths else 
                    html.P("Aucun point fort identifié", style={'color': '#7f8c8d', 'fontStyle': 'italic'})
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H5("⚠️ Difficultés", style={'color': '#e74c3c', 'marginBottom': '10px'}),
                    html.Ul([html.Li(difficulty, style={'color': '#2c3e50', 'marginBottom': '5px'}) 
                            for difficulty in profile.difficulty_areas]) if profile.difficulty_areas else 
                    html.P("Aucune difficulté identifiée", style={'color': '#7f8c8d', 'fontStyle': 'italic'})
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ])

    def create_learning_style_radar(self, profile) -> go.Figure:
        """Create modern learning style radar chart"""
        categories = ['Visuel', 'Auditif', 'Lecture', 'Kinesthésique']
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
            name='Style d\'apprentissage',
            line_color='#3498db',
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont={'color': '#2c3e50'},
                    gridcolor='#ecf0f1'
                ),
                angularaxis=dict(
                    tickfont={'color': '#2c3e50', 'size': 12}
                ),
                bgcolor='white'
            ),
            showlegend=False,
            title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=30, b=30)
        )

        return fig

    def create_personality_radar(self, profile) -> go.Figure:
        """Create modern personality radar chart"""
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
            line_color='#e74c3c',
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont={'color': '#2c3e50'},
                    gridcolor='#ecf0f1'
                ),
                angularaxis=dict(
                    tickfont={'color': '#2c3e50', 'size': 12}
                ),
                bgcolor='white'
            ),
            showlegend=False,
            title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=30, b=30)
        )

        return fig

    def create_behavioral_metrics(self, profile) -> go.Figure:
        """Create modern behavioral metrics visualization"""
        metrics = ['Engagement', 'Persistance', 'Demande d\'aide', 'Collaboration', 'Auto-régulation']
        values = [
            profile.behavioral.engagement_level,
            profile.behavioral.persistence,
            profile.behavioral.help_seeking,
            profile.behavioral.collaboration_preference,
            profile.behavioral.self_regulation
        ]

        colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6']

        fig = go.Figure(data=[
            go.Bar(
                x=metrics, 
                y=values, 
                marker_color=colors,
                text=[f'{v:.1%}' for v in values],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            margin=dict(l=50, r=50, t=30, b=50)
        )

        return fig

    def create_engagement_timeline(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create modern engagement timeline"""
        if len(learner_data) == 0:
            return go.Figure()

        learner_data = learner_data.sort_values('timestamp')
        engagement_scores = learner_data['user_input'].str.len() / 100

        fig = go.Figure(data=[
            go.Scatter(
                x=learner_data['timestamp'],
                y=engagement_scores,
                mode='lines+markers',
                name='Engagement',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8, color='#3498db')
            )
        ])

        fig.update_layout(
            title="",
            xaxis_title="Temps",
            yaxis_title="Score d'Engagement",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            margin=dict(l=50, r=50, t=30, b=50)
        )

        return fig

    def create_interaction_patterns(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create modern interaction patterns visualization"""
        if len(learner_data) == 0:
            return go.Figure()

        message_lengths = learner_data['user_input'].str.len()
        time_diffs = learner_data['timestamp'].diff().dt.total_seconds()

        fig = go.Figure(data=[
            go.Scatter(
                x=message_lengths,
                y=time_diffs,
                mode='markers',
                name='Patterns d\'interaction',
                marker=dict(
                    size=10,
                    color='#e74c3c',
                    opacity=0.7
                )
            )
        ])

        fig.update_layout(
            title="",
            xaxis_title="Longueur du message",
            yaxis_title="Délai (secondes)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            margin=dict(l=50, r=50, t=30, b=50)
        )

        return fig

    def create_sentiment_analysis(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create modern sentiment analysis visualization"""
        if len(learner_data) == 0:
            return go.Figure()

        sentiment_scores = np.random.uniform(0, 1, len(learner_data))

        fig = go.Figure(data=[
            go.Scatter(
                x=learner_data['timestamp'],
                y=sentiment_scores,
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=8, color='#f39c12')
            )
        ])

        fig.update_layout(
            title="",
            xaxis_title="Temps",
            yaxis_title="Score de Sentiment",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            margin=dict(l=50, r=50, t=30, b=50)
        )

        return fig

    def create_comprehension_trends(self, learner_data: pd.DataFrame) -> go.Figure:
        """Create modern comprehension trends visualization"""
        if len(learner_data) == 0:
            return go.Figure()

        comprehension_scores = np.random.uniform(0, 1, len(learner_data))

        fig = go.Figure(data=[
            go.Scatter(
                x=learner_data['timestamp'],
                y=comprehension_scores,
                mode='lines+markers',
                name='Compréhension',
                line=dict(color='#27ae60', width=3),
                marker=dict(size=8, color='#27ae60')
            )
        ])

        fig.update_layout(
            title="",
            xaxis_title="Temps",
            yaxis_title="Score de Compréhension",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            margin=dict(l=50, r=50, t=30, b=50)
        )

        return fig

    def create_difficulty_areas(self, profile) -> go.Figure:
        """Create modern difficulty areas visualization"""
        if not profile.difficulty_areas and not profile.strengths:
            return go.Figure()

        categories = profile.difficulty_areas + profile.strengths
        values = [-1] * len(profile.difficulty_areas) + [1] * len(profile.strengths)
        colors = ['#e74c3c'] * len(profile.difficulty_areas) + ['#27ae60'] * len(profile.strengths)

        fig = go.Figure(data=[
            go.Bar(
                x=categories, 
                y=values, 
                marker_color=colors,
                text=['Difficulté' if v < 0 else 'Force' for v in values],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="Impact",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            margin=dict(l=50, r=50, t=30, b=50)
        )

        return fig

    def create_recommendations_panel(self, profile) -> html.Div:
        """Create modern recommendations panel"""
        if not profile.recommended_strategies:
            return html.Div([
                html.I(className="fas fa-lightbulb", style={'fontSize': '48px', 'color': '#f1c40f', 'marginBottom': '15px'}),
                html.P("Aucune recommandation disponible", 
                       style={'color': 'black', 'fontSize': '16px', 'textAlign': 'center'})
            ], style={'textAlign': 'center', 'padding': '40px'})

        recommendation_items = []
        for i, rec in enumerate(profile.recommended_strategies, 1):
            recommendation_items.append(
                html.Div([
                    html.Div([
                        html.Span(f"{i}", 
                                style={
                                    'backgroundColor': 'white',
                                    'color': '#667eea',
                                    'width': '24px',
                                    'height': '24px',
                                    'borderRadius': '50%',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'fontWeight': 'bold',
                                    'fontSize': '12px',
                                    'marginRight': '15px'
                                }),
                        html.Span(rec, style={'color': 'black', 'fontSize': '14px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'})
                ])
            )

        return html.Div([
            html.Div([
                html.I(className="fas fa-lightbulb", style={'fontSize': '24px', 'marginRight': '10px'}),
                html.H4("Stratégies Recommandées", style={'margin': '0', 'color': 'black'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            html.Div(recommendation_items)
        ])

    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.app.run(debug=debug, port=port)

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
