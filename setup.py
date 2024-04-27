# stdlib
import os

# third party
from setuptools import setup

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version() -> str:
    return "1.0.0"


if __name__ == "__main__":
    try:
        setup(
            name="crnsynth",
            version=get_version(),
            description="Synthetic data generation methods for the Cancer Registry of Norway",
            author="Daan Knoors, Severin Elvatun",
            author_email="dk@daanknoors.com, sela@kreftregisteret.no",
            long_description=read("README.md"),
            install_requires=[
                "synthetic-data-generation>=0.1.14",
                "synthcity",
                "diffprivlib>=0.6.3",
                "sdmetrics",
                "gower",
            ],
            extras_require={"interactive": ["matplotlib", "jupyter"]},
        )
    except:  
        # error 
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
