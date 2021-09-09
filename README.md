# geohash-modelling
Geographic anomaly detection based on latitude/longitudes using the geohash algorithm. 

This method is useful because raw clustering based on latitudes/longitudes ignores that they are truly a sphere, making clustering less accurate. Projecting 2d lat/lon data to 3d using havershine distance is also an inaccurate approach prone to error, which we will show by example.
