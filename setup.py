##started the show
from setuptools import setup, find_packages

setup(
    name="voiceauthCore",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow", "librosa", "pydub", "numpy", "scipy", "transformers"
    ],
    entry_points={
        "console_scripts": [
            "voiceauthCore=voiceauthCore.core:main"
        ]
    }
)
