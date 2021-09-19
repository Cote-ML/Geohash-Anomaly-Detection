import setuptools

setuptools.setup(
    name="GeoHashCluster",
    version="0.0.1",
    author="Lee Cote",
    author_email="brian.lee.cote@gmail.com",
    description="lorem ipsum",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    python_requires=">=3.6",
)