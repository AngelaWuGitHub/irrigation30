# `irrigation30` - Python3 Package

**irrigation30** is a Python package designed to help researchers predict and visualize irrigation at 30m resolution for small regions at a time using Google Earth Engine.

With this package, users can generate and visualize irrigation predictions with an interactive map powered by Google Earth Engine and `folium`. The package also includes additional features to analyze relevant climate and satellite data like temperature, precipitation, NDVI signatures, and satellite images from Sentinel-2.

This project was developed by a team of UC Berkeley students in the Master’s in Information and Data Science program.

**Website**: https://groups.ischool.berkeley.edu/irrigation30/webdev/


## Installation

### Dependencies

`irrigation30` requires:
- Python (>= 3.6)
- pandas (>= 1.0.5)
- numpy (>= 1.19.0)
- scipy (>= 1.5.0)
- matplotlib (>= 3.2.2)
- earthengine-api (>= 0.1.227)
- folium (>= 0.11.0)
- shapely (>= 1.7.0)

Our package also requires users to have a Google Earth Engine account. You can sign up for an account [here](https://signup.earthengine.google.com/#!/). Note that it may take several days for Google to activate your account.

Lastly, our package is intended to be used in Python Jupyter notebooks. For more instructions on how to install Jupyter and use Jupyter notebooks, see [this tutorial](https://realpython.com/jupyter-notebook-introduction/).

## User installation
The easiest way to install `irrigation30` is with `pip3`
```
pip3 install irrigation30
```

If you are having a hard time getting the right package versions installed, you can set up a virtual environment as follows:

##### Install with `virtualenv`
1. `pip3 install virtualenv`
2. `virtualenv env`
3. `source env/bin/activate`
4. `curl https://bootstrap.pypa.io/get-pip.py | python3`
5. `pip3 install irrigation30`
6. `pip3 install ipykernel`
7. `ipython kernel install --user --name=env`
8. `jupyter notebook`
9. When finished, deactivate your virtualenv with `deactivate`


## Usage

For a full tutorial of how the package is used in Jupyter notebooks, see our Jupyter notebook demo `Demo.ipynb`.

For more info on how to download TIF assets to Google Earth Engine and download TIF files to Google Drive, see `docs/download_TIF.md`.

## Help and Support

For help with using or installing the package, contact Weixin ("Angela") Wu: https://www.ischool.berkeley.edu/people/weixin-wu
