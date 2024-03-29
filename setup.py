from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scopes",
    version="0.2.0",
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
        "astropy",
        "astroplan",
        "tqdm",
        "pytz",
        "timezonefinder",
    ],
    extras_require={"dev": ["pytest", "sphinx"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="GNU General Public License v3.0",
    keywords="astronomy scheduling observation planning",
)
