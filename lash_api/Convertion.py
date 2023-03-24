import pyproj
import numpy as np
from pyproj import CRS

class Convert:
    def get_lat_long_to_meters(self, flid,X_list1, Y_list1, X_list2, Y_list2):
        distance = []
        wgs_proj = CRS.from_string("WGS 84")
        epsg_proj =CRS.from_string("EPSG:4326")  # Stays in native unit
        for i in range(len(X_list1)):
            x1, y1 = pyproj.transform(wgs_proj, epsg_proj, X_list1[i], Y_list1[i])
            x2, y2 = pyproj.transform(wgs_proj, epsg_proj, X_list2[i], Y_list2[i])
            distance_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance.insert(0, (distance_m[0],flid[i]))
        return distance

    def get_lat_long_to_meters2(self, flid,X_list1, Y_list1, X_list2, Y_list2):
        distance = []
        wgs_proj = CRS.from_string("WGS 84")
        epsg_proj =CRS.from_string("EPSG:4326")  # Stays in native unit
        for i in range(len(X_list1)):
            x1, y1 = pyproj.transform(wgs_proj, epsg_proj, X_list1[i], Y_list1[i])
            x2, y2 = pyproj.transform(wgs_proj, epsg_proj, X_list2[i], Y_list2[i])
            distance_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance.insert(0, (distance_m,flid[i]))
        return distance
