from setuptools import setup, find_packages

try:
    with open("README.md", encoding="utf8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A deepfake audio detection tool"

setup(
    name="voiceauthCore",
    version="0.1.51",
    author="Sadiq Kassamali",
    author_email="sadiq.kasssamali@gmail.com",
    description="A deepfake audio detection tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sadiqkassamali/voiceauthCore",
    package_dir={"voiceauthCore": "src/voiceauthCore"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "librosa",
        "pydub",
        "numpy",
        "scipy",
        "transformers",
        "pillow",
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
