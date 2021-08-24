import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ParticlePointTracker", # Replace with your own username
    version="0.0.1",
    author="Felix Lehner",
    author_email="info@felixlehner.de",
    description="A simple particle point tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Centrasis/pyParticlePointTracker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires = [
        "numpy",
        "opencv-python",
        "joblib",
        "scikit-image"
    ]
)