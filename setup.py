from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="driver-drowsiness-detection",
    version="1.0.0",
    author="Driver Safety Systems",
    author_email="contact@driversafety.com",
    description="Real-time driver drowsiness detection system for Raspberry Pi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/driversafety/driver-drowsiness-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "drowsiness-detector=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.h5"],
    },
    data_files=[
        ("config", ["config.yaml"]),
        ("models", ["models/drowsiness_cnn.h5", "models/eye_state_vit.h5"]),
    ],
    platforms=["linux"],
    keywords="drowsiness detection raspberry pi computer vision deep learning",
    project_urls={
        "Bug Reports": "https://github.com/LeenaKannan/Real-Time-Driver-Drowsiness-Detection/issues",
        "Source": "https://github.com/LeenaKannan/Real-Time-Driver-Drowsiness-Detection",
        "Documentation": "https://github.com/LeenaKannan/Real-Time-Driver-Drowsiness-Detection/wiki",
    },
)