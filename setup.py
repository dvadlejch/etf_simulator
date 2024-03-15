from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "ETF Simulator"
LONG_DESCRIPTION = "A tool for data processing and statistical predictions of ETFs"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="etf_simulator",
    version=VERSION,
    author="Daniel Vadlejch",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "etf"],
)
