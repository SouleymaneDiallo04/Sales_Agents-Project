from setuptools import setup, find_packages

setup(
    name="sales_agents_project",
    version="1.0.0",
    description="RL Pricing System for E-commerce",
    author="Your Team",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "torch>=2.0.0",
        "mysql-connector-python>=8.1.0",
        "fastapi>=0.104.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rl-pricing=scripts.run_pricing_cycle:main",
            "rl-train=scripts.fine_tune_sales:main",
            "rl-api=deployment.api.fastapi_app:main",
        ]
    },
)