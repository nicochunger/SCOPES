import re

from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read version directly from the package __init__.py file
with open("scopes/__init__.py", "r") as f:
    version = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
    name="scopes-astro",
    version=version,
    author="Nicolas Unger",
    author_email="nicolas.unger@unige.ch",
    description="System for Coordinating Observational Planning and Efficient Scheduling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_dir={"scopes": "scopes"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "astropy",
        "astroplan",
        "tqdm",
        "pytz",
        "timezonefinder",
    ],
    extras_require={"dev": ["pytest", "sphinx"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="GNU General Public License v3.0",
    keywords="astronomy scheduling observation planning",
    project_urls={
        "GitHub": "https://github.com/nicochunger/SCOPES",
        # "Documentation": "https://scopes-astro.readthedocs.io/en/latest/",
        "Documentation": "https://github.com/nicochunger/SCOPES/blob/main/SCOPES_documentation.pdf",
        "Changelog": "https://github.com/nicochunger/SCOPES/blob/main/CHANGELOG.txt",
    },
)
