## started the show
from setuptools import setup, find_packages

setup(
    name="voiceauthCore",
    version="0.6.5.1",
    author="sadiq kassamali",
    author_email="sadiq.kasssamali@gmail.com",
    description="A deepfake audio detection tool",
    long_description=open('README.md',encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sadiqkassamali/voiceauthCore",
    packages=find_packages(),
    install_requires=[
        "tensorflow", "librosa", "pydub", "numpy", "scipy", "transformers" , "torch", "pillow"
    ],
    entry_points={
        "console_scripts": [
            "voiceauthCore=voiceauthCore.core:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)