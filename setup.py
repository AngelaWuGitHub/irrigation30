import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="irrigation30",
    version="0.0.2",
    author="Weixin Wu",
    author_email="wwu34@berkeley.edu",
    description="Generate irrigation predictions at 30m resolution using Google Earth Engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AngelaWuGitHub/irrigation30",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.0.5',
        'shapely>=1.7.0',
        'earthengine-api>=0.1.227',
        'folium>=0.11.0',
        'matplotlib>=3.2.2',
        'numpy>=1.19.0',
        'scipy>=1.5.0'
    ]
)
