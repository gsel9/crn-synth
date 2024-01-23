from setuptools import find_packages, setup

setup(
    name="crnsynth",
    version="0.0.1",
    description="Synthetic data generation methods for the Cancer Registry of Norway",
    author="Daan Knoors, Severin Elvatun",
    author_email="dk@daanknoors.com, sela@kreftregisteret.no",
    packages=find_packages(),
    install_requires=[
        "synthetic-data-generation>=0.1.14",
        "synthcity",
        "diffprivlib>=0.6.3",
        "sdmetrics",
        "gower",
    ],
    extras_require={"interactive": ["matplotlib", "jupyter"]},
)
