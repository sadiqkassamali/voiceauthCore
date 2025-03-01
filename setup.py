## started the show
from setuptools import setup, find_packages

try:
    with open("README.md", encoding="utf8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A deepfake audio detection tool"

setup(
    name="voiceauthCore",
    version="0.1.1",  # Use proper semantic versioning
    author="Sadiq Kassamali",
    author_email="sadiq.kasssamali@gmail.com",
    description="A deepfake audio detection tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sadiqkassamali/voiceauthCore",
    package_dir={"": "src"},  # Look for packages inside src/
    packages=find_packages(where="src"),  # Find packages inside src/
    include_package_data=True,  # Ensures all package data is included
    install_requires=[
        "tensorflow",
        "librosa",
        "pydub",
        "numpy",
        "scipy",
        "transformers",
        "pillow",
        # ⚠️ Torch should be installed separately to avoid issues
        "torch ; sys_platform != 'darwin' and python_version >= '3.6'",
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
    python_requires=">=3.6",
)
