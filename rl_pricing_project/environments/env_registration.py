from gymnasium.envs.registration import register

register(
    id='EcommercePricing-v1',
    entry_point='environments.pricing_env:EcommercePricingEnv',
    max_episode_steps=365,
    reward_threshold=2000.0,
    nondeterministic=True,
)

register(
    id='EcommercePricing-v2',
    entry_point='environments.pricing_env:EcommercePricingEnv',
    max_episode_steps=180,  # 6 mois
    reward_threshold=1000.0,
    nondeterministic=True,
    kwargs={'product_id': 'PROD_001'}
)

def register_envs():
    """Fonction utilitaire pour enregistrer tous les environnements"""
    print("✅ Environnements EcommercePricing enregistrés pour SB3 Zoo")