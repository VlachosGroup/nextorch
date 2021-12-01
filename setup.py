#
# setup.py
#
# Installation script to get setuptools to install nextorch into
# a Python environment.
#
import os
import sys
import setuptools

# Import the lengthy rich-text README as the package's long
# description:
root_dir = os.path.dirname(__file__)

with open(os.path.join(root_dir, "README.rst"), "r") as fh:
	long_description = fh.read()


setuptools.setup(
    name="nextorch", 
    version="0.1.0",
    author="Vlachos Research Group",
    author_email="vlachos@udel.edu",
    description="Experimental design and Bayesian optimization library in Python/PyTorch",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/VlachosGroup/nextorch",
    project_urls={
        "Documentation": "https://nextorch.readthedocs.io/en/latest/",
        "Source": "https://github.com/VlachosGroup/nextorch",
    },
    packages=setuptools.find_packages(),
    package_data={'':['*.xlsx']},
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8", 
        "gpytorch>=1.4",
        "botorch>=0.4.0",
        "matplotlib>=3.1.1",
        "pyDOE2>=1.3.0",
        "numpy>=1.19.2",
        "scipy>=1.3.1",
        "pandas>=0.25.1",
        "openpyxl>=3.0.7",
        "pytest>=6.2.3"],
    classifiers=[
        "Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Chemistry",
    ],
)