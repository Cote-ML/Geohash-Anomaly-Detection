import setuptools

setuptools.setup(
    name="GeoHashCluster",
    version="1.0.0",
    author="Lee Cote",
    description="Location anomaly detection using a 3-dim normal distribution fits along with the geoHash algorithm.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    python_requires=">=3.6",
)