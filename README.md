# trackingML
Machine learning for tracking algorithms


# Installation

I intend to use virtual environement for the python development. 
First download miniconda from [link](https://conda.io/miniconda.html),
and install it to $HOME directory.
Then create the environment from YML file
```
conda env create -f environment.yml
```
It should install most of necessary packages.

Then download the [trackml-library](https://github.com/LAL/trackml-library):
```
git clone https://github.com/LAL/trackml-library.git
```
and install *trackml-library*
```
cd trackml-library
pip install --user .
```
