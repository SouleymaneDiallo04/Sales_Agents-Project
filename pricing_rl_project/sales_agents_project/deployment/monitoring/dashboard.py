"""
Dashboard PPO - Site de vente avec IA
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# Configuration avec thème blanc propre
st.set_page_config(
    page_title="IA Pricing - Dashboard Commercial",
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour interface blanche épurée
st.markdown("""
<style>
    /* Fond blanc propre */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Titres simples */
    .section-title {
        color: #2E8B57;
        padding: 10px 0;
        margin: 15px 0;
        font-weight: bold;
        border-bottom: 3px solid #2E8B57;
        font-size: 1.4em;
    }
    
    /* Cartes légères */
    .light-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Boutons verts */
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #3CB371;
    }
    
    /* Métriques */
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>💰 Plateforme IA de Pricing Intelligent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Optimisez vos prix avec notre intelligence artificielle avancée</p>", unsafe_allow_html=True)

# ============================================
# 1. ÉTAT DU MODÈLE IA
# ============================================

st.markdown("<div class='section-title'>📊 État du Système IA</div>", unsafe_allow_html=True)

# VRAI chemin de ton modèle
CHEMIN_MODELE = r"C:\Users\Alif computer\Desktop\projet machine learning\pricing_rl_project\sales_agents_project\pretrained_models\custom_trained\best_PROD_001.zip"

# Vérifier si le modèle existe
if os.path.exists(CHEMIN_MODELE):
    col1, col2 = st.columns(2)  # SEULEMENT 2 colonnes au lieu de 3
    
    with col1:
        st.markdown("<div class='light-card'>", unsafe_allow_html=True)
        st.metric("État du modèle", "Actif", delta="En ligne")
        st.caption("best_PROD_001.zip")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='light-card'>", unsafe_allow_html=True)
        date_modif = datetime.fromtimestamp(os.path.getmtime(CHEMIN_MODELE))
        st.metric("Dernière mise à jour", date_modif.strftime("%d/%m %H:%M"))
        st.caption("Système à jour")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Essayer de charger le modèle
    try:
        from stable_baselines3 import PPO
        
        # Charger le modèle
        modele = PPO.load(CHEMIN_MODELE)
        
        # Sauvegarder dans session
        st.session_state.modele_ppo = modele
        st.session_state.modele_charge = True
        
        st.success("✅ Modèle IA chargé avec succès")
        
    except Exception as e:
        st.error(f"❌ Erreur technique: {str(e)[:100]}...")
        st.session_state.modele_charge = False
        st.info("Mode simulation activé")
        
else:
    st.error("❌ Modèle introuvable")
    st.session_state.modele_charge = False

# ============================================
# 2. SIMULATEUR COMMERCIAL
# ============================================

class SimulateurCommercial:
    """Simulateur commercial pour optimisation des prix"""
    
    def __init__(self):
        self.produits = {
            "PROD_001": {
                "nom": "Ordinateur Gaming Pro", 
                "prix_base": 999.99, 
                "cost": 600.00,
                "stock": 45,
                "marge_min": 0.15,
                "ventes_mensuelles": 120
            },
            "PROD_002": {
                "nom": "Smartphone Élite", 
                "prix_base": 799.99, 
                "cost": 450.00,
                "stock": 120,
                "marge_min": 0.20,
                "ventes_mensuelles": 300
            },
            "PROD_003": {
                "nom": "Casque Audio Premium", 
                "prix_base": 199.99, 
                "cost": 80.00,
                "stock": 200,
                "marge_min": 0.25,
                "ventes_mensuelles": 500
            }
        }
        
        self.clients = {
            "Premium": {"budget": 1500, "sensibilite": 0.3, "fidelite": 0.8, "taux_conversion": 0.7},
            "Standard": {"budget": 1000, "sensibilite": 0.5, "fidelite": 0.6, "taux_conversion": 0.5},
            "Économique": {"budget": 500, "sensibilite": 0.8, "fidelite": 0.4, "taux_conversion": 0.3},
            "Entreprise": {"budget": 2000, "sensibilite": 0.4, "fidelite": 0.7, "taux_conversion": 0.6}
        }
    
    def preparer_etat(self, produit_id, client_type, contexte):
        """Prépare l'état pour l'IA"""
        produit = self.produits[produit_id]
        client = self.clients[client_type]
        
        # Créer état
        etat = np.zeros(10, dtype=np.float32)
        
        # Features produit
        etat[0] = produit["stock"] / 250.0
        etat[1] = produit["prix_base"] / 2000.0
        etat[2] = produit["cost"] / produit["prix_base"] if produit["prix_base"] > 0 else 0.6
        etat[3] = produit["marge_min"]
        
        # Features client
        etat[4] = client["sensibilite"]
        etat[5] = client["fidelite"]
        etat[6] = client["budget"] / 2500.0
        
        # Features marché
        etat[7] = contexte.get("competition", 0.5)
        etat[8] = contexte.get("demand", 0.7)
        etat[9] = 1.0 if datetime.now().weekday() >= 5 else 0.0
        
        return etat
    
    def interpreter_action(self, action, prix_base, produit_id):
        """Interprète la décision de l'IA"""
        actions = {
            0: {"nom": "Promotion forte", "change": -0.15, "strategie": "Liquidité & Volume"},
            1: {"nom": "Promotion", "change": -0.07, "strategie": "Compétitivité"},
            2: {"nom": "Stabilité", "change": 0.00, "strategie": "Fidélisation"},
            3: {"nom": "Augmentation", "change": 0.07, "strategie": "Marge optimisée"},
            4: {"nom": "Premium", "change": 0.15, "strategie": "Positionnement"}
        }
        
        action_info = actions.get(action, actions[2])
        
        nouveau_prix = prix_base * (1 + action_info["change"])
        nouveau_prix = max(nouveau_prix, prix_base * 0.5)
        
        # Calcul bénéfice estimé
        produit = self.produits[produit_id]
        benefice_unite = nouveau_prix - produit["cost"]
        benefice_estime = benefice_unite * produit["ventes_mensuelles"] * (1 - abs(action_info["change"]))
        
        return {
            "action_id": action,
            "action_nom": action_info["nom"],
            "change_percent": action_info["change"] * 100,
            "prix_final": round(nouveau_prix, 2),
            "strategie": action_info["strategie"],
            "benefice_unite": round(benefice_unite, 2),
            "benefice_estime": round(benefice_estime, 2)
        }

# Initialiser simulateur
simulateur = SimulateurCommercial()

# ============================================
# 3. INTERFACE COMMERCIALE
# ============================================

st.markdown("<div class='section-title'>📈 Analyse de Marché</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    st.markdown("### Configuration")
    
    # Sélection produit
    produit_options = {k: f"{k} - {v['nom']}" for k, v in simulateur.produits.items()}
    produit_select = st.selectbox(
        "Produit à analyser",
        list(produit_options.keys()),
        format_func=lambda x: produit_options[x]
    )
    
    # Sélection client
    client_select = st.selectbox(
        "Segment client",
        list(simulateur.clients.keys())
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)  # UNIQUEMENT 2 cartes dans la sidebar
    st.markdown("### Environnement marché")
    
    competition = st.slider(
        "Intensité concurrentielle",
        0.0, 1.0, 0.5, 0.1,
        help="Niveau de compétition sur le marché"
    )
    
    demande = st.slider(
        "Niveau de demande",
        0.0, 1.0, 0.7, 0.1,
        help="Demande globale du marché"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Boutons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Lancer l'analyse", use_container_width=True):
            st.session_state.demande_decision = True
    
    with col2:
        if st.button("Nouvelle session", use_container_width=True):
            if 'decisions' in st.session_state:
                st.session_state.decisions = []
            st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # ANALYSE EN TEMPS RÉEL
    if st.session_state.get('demande_decision', False):
        # Préparer contexte
        contexte = {
            "competition": competition,
            "demand": demande,
            "timestamp": datetime.now()
        }
        
        # Préparer état
        produit = simulateur.produits[produit_select]
        etat = simulateur.preparer_etat(produit_select, client_select, contexte)
        
        # Obtenir décision
        if st.session_state.get('modele_charge', False):
            try:
                modele = st.session_state.modele_ppo
                action, _ = modele.predict(etat, deterministic=True)
                action_id = int(action[0])
                resultat = simulateur.interpreter_action(action_id, produit["prix_base"], produit_select)
                mode_ia = True
            except:
                action_id = np.random.choice([0, 1, 2, 3, 4])
                resultat = simulateur.interpreter_action(action_id, produit["prix_base"], produit_select)
                mode_ia = False
        else:
            action_id = np.random.choice([0, 1, 2, 3, 4])
            resultat = simulateur.interpreter_action(action_id, produit["prix_base"], produit_select)
            mode_ia = False
        
        # Créer décision
        decision = {
            "timestamp": datetime.now(),
            "produit_id": produit_select,
            "produit_nom": produit["nom"],
            "client_type": client_select,
            "prix_base": produit["prix_base"],
            "prix_final": resultat["prix_final"],
            "changement_percent": resultat["change_percent"],
            "action_id": action_id,
            "action_nom": resultat["action_nom"],
            "strategie": resultat["strategie"],
            "benefice_unite": resultat["benefice_unite"],
            "benefice_estime": resultat["benefice_estime"],
            "etat_features": etat.tolist(),
            "mode_ia": mode_ia
        }
        
        # Sauvegarder
        if 'decisions' not in st.session_state:
            st.session_state.decisions = []
        
        st.session_state.decisions.append(decision)
        st.session_state.demande_decision = False
        
        # Afficher résultat
        st.markdown("<div class='light-card'>", unsafe_allow_html=True)
        st.markdown("### Résultat de l'analyse IA")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Produit", produit['nom'])
            st.metric("Prix actuel", f"€{produit['prix_base']:.2f}")
        
        with col_b:
            st.metric("Segment", client_select)
            st.metric("Prix optimisé", f"€{resultat['prix_final']:.2f}")
        
        # Détails
        st.markdown("---")
        
        st.markdown(f"""
        **Stratégie :** {resultat['action_nom']} ({resultat['change_percent']:+.1f}%)  
        **Bénéfice unitaire :** €{resultat['benefice_unite']:.2f}  
        **Bénéfice mensuel estimé :** €{resultat['benefice_estime']:,.0f}  
        **Source :** {'IA avancée' if mode_ia else 'Simulation'}
        """)
        
        # Visualisation
        if resultat["change_percent"] > 0:
            st.progress(min(resultat["change_percent"] / 15, 1.0), "Hausse recommandée")
        elif resultat["change_percent"] < 0:
            st.progress(min(abs(resultat["change_percent"]) / 15, 1.0), "Promotion recommandée")
        else:
            st.progress(0.5, "Stabilité recommandée")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.markdown("<div class='light-card'>", unsafe_allow_html=True)
        st.info("Utilisez le panneau de configuration pour lancer une analyse")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    st.markdown("### Indicateurs")
    
    # Stats session
    if 'decisions' in st.session_state and st.session_state.decisions:
        df = pd.DataFrame(st.session_state.decisions)
        
        if not df.empty:
            # Distribution stratégies
            st.markdown("**Répartition des stratégies**")
            action_counts = df['action_nom'].value_counts()
            st.bar_chart(action_counts)
            
            # Métriques
            avg_change = df['changement_percent'].mean()
            st.metric("Variation moyenne", f"{avg_change:+.1f}%")
            st.metric("Analyses", len(df))
    
    # Info modèle
    st.markdown("---")
    st.markdown("**État du système**")
    
    if st.session_state.get('modele_charge', False):
        st.success("✅ IA opérationnelle")
    else:
        st.warning("⚠️ Mode simulation")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 4. HISTORIQUE & ANALYSE
# ============================================

if 'decisions' in st.session_state and st.session_state.decisions:
    st.markdown("<div class='section-title'>📜 Historique des Analyses</div>", unsafe_allow_html=True)
    
    df = pd.DataFrame(st.session_state.decisions)
    
    # Formater affichage
    df_display = df.copy()
    df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%H:%M:%S')
    df_display['prix_base'] = df_display['prix_base'].apply(lambda x: f"€{x:.2f}")
    df_display['prix_final'] = df_display['prix_final'].apply(lambda x: f"€{x:.2f}")
    df_display['changement_percent'] = df_display['changement_percent'].apply(lambda x: f"{x:+.1f}%")
    df_display['benefice_estime'] = df_display['benefice_estime'].apply(lambda x: f"€{x:,.0f}")
    
    # Afficher tableau
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    st.dataframe(
        df_display[['timestamp', 'produit_nom', 'client_type', 'prix_base', 
                   'prix_final', 'changement_percent', 'benefice_estime', 'strategie']],
        column_config={
            "timestamp": "Heure",
            "produit_nom": "Produit",
            "client_type": "Segment",
            "prix_base": "Prix initial",
            "prix_final": "Prix optimisé",
            "changement_percent": "Variation",
            "benefice_estime": "Bénéfice estimé",
            "strategie": "Stratégie"
        },
        hide_index=True,
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Graphiques
    if len(df) > 1:
        st.markdown("<div class='section-title'>📈 Visualisation</div>", unsafe_allow_html=True)
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("<div class='light-card'>", unsafe_allow_html=True)
            st.markdown("**Évolution des prix**")
            
            # Graphique prix
            df_graph = df.copy()
            df_graph['index'] = range(len(df_graph))
            
            import plotly.graph_objects as go
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_graph['index'],
                y=df_graph['prix_base'],
                name='Prix initial',
                line=dict(color='blue', dash='dash')
            ))
            fig1.add_trace(go.Scatter(
                x=df_graph['index'],
                y=df_graph['prix_final'],
                name='Prix optimisé',
                line=dict(color='green')
            ))
            fig1.update_layout(
                xaxis_title='Analyse #',
                yaxis_title='Prix (€)',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_g2:
            st.markdown("<div class='light-card'>", unsafe_allow_html=True)
            st.markdown("**Évolution du bénéfice**")
            
            # Graphique bénéfice
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_graph['index'],
                y=df_graph['benefice_estime'],
                name='Bénéfice estimé',
                line=dict(color='darkgreen', width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 139, 87, 0.2)'
            ))
            fig2.update_layout(
                xaxis_title='Analyse #',
                yaxis_title='Bénéfice (€)',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Graphique cumulatif bénéfice
        st.markdown("<div class='light-card'>", unsafe_allow_html=True)
        st.markdown("**Bénéfice cumulé**")
        
        df['benefice_cumul'] = df['benefice_estime'].cumsum()
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['benefice_cumul'],
            name='Bénéfice cumulé',
            line=dict(color='#2E8B57', width=4),
            fill='tozeroy',
            fillcolor='rgba(46, 139, 87, 0.3)'
        ))
        fig3.update_layout(
            xaxis_title='Temps',
            yaxis_title='Bénéfice cumulé (€)',
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 5. FOOTER SIMPLE
# ============================================

st.markdown("---")

col1, col2 = st.columns(2)  # SEULEMENT 2 colonnes au lieu de 3

with col1:
    st.markdown("**Plateforme IA Pricing**  \nOptimisation intelligente des prix")

with col2:
    current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    st.markdown(f"**Session active**  \n{current_time}")

st.markdown("<p style='text-align: center; color: #666; font-size: 0.8em; margin-top: 20px;'>Système d'intelligence artificielle pour l'optimisation commerciale</p>", unsafe_allow_html=True)

# Mode auto-refresh
if st.session_state.get('auto_refresh', False):
    time.sleep(5)
    st.rerun()