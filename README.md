# geohash-modelling
Geographic anomaly detection based on latitude/longitudes using the geohash algorithm. 

This method is useful because raw clustering based on latitudes/longitudes ignores that they are truly a sphere, making clustering less accurate. Projecting 2d lat/lon data to 3d using havershine distance is also an inaccurate approach prone to error. Projecting geolocations onto a 64x64/128x128 hash map of the world and applying multivariate normals to the centroids of each hash allows us to seed a probability matrix based on a customer's legitimate geographic profile and then likewise assign a probability of outlierness. 

Fascinatingly, you can also apply this in a non-anomaly detection methodology. If you've a dataset of geographic points, and instead approach it as a classical DBSCAN clustering problem, the hash algorithm will intelligently discern geograhpic regions on its own-- e.g., "Eastern vs Midwestern vs Western U.S.", "Eastern vs Western Europe", etc. 
