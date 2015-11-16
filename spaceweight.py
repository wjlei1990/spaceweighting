#!/usr/bin/env python
# mimicing the voronoi

from __future__ import print_function

import os
import glob
import numpy as np
from obspy.geodetics import locations2degrees
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import collections

from spherical_voronoi import SphericalVoronoi                                  
from matplotlib import colors                                                   
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                         
from mpl_toolkits.mplot3d import proj3d                                         
import scipy.spatial                                                            
import math
from math import sin, cos
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import Beach

import operator
from pprint import pprint


class Station(object):

    def __init__(self, network="", station="", latitude=None, longitude=None,
                 elevation=None):
        self.network = network
        self.station = station
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.tag = "%s_%s" % (self.network, self.station)

        self._sanity_check()

    def _sanity_check(self):
        if not isinstance(self.network, str):
            raise ValueError("Network should be a string: %s" 
                             % str(self.network))
        if not isinstance(self.station, str):
            raise ValueError("Station should be a string: %s" 
                             % str(self.station))
        if self.latitude < -90.0 or self.latitude > 90.0:
            raise ValueError("Latitude should be between [-90, 90]: %f" 
                             % self.latitude)
        if self.longitude < -180.0 or self.longitude > 180.0:
            raise ValueError("Longitude should be between [-180, 180]: %f" 
                             % self.longitude)

    def __str__(self):
        return "[network:%-3s, station:%-5s, (lat=%-7.2f, lon=%-7.2f, " \
               "elev=%-7.2f)]" \
               % (self.network, self.station, self.latitude, 
                  self.longitude, self.elevation)


class Event(object):
    
    def __init__(self, latitude=None, longitude=None):
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return "[Event: (latitude=%-7.2f, longitude=%-7.2f)]" \
                % (self.latitude, self.longitude)


class WeightBase(object):
    
    def __init__(self, stations, event):

        self.stations = stations
        self._stations = stations
        self.event = event

        self.station_tag, self.station_loc, self.network_dict = \
                self._sort_and_check_stations()
        self.nstations = len(self.stations)

        self.weight = np.zeros(self.nstations)

    def _find_duplicate(self):
        sta_tag = []
        sta_loc = []
        for station in self.stations:
            sta_tag.append(station.tag)
            sta_loc.append([station.latitude, station.longitude])
        sta_loc = np.array(sta_loc)
        # check station name
        if len(sta_tag) != len(set(sta_tag)):
            raise ValueError("Duplicate station names!")
        # check station location
        dim = sta_loc.shape[0]
        b = np.ascontiguousarray(sta_loc).view(np.dtype(
            (np.void, sta_loc.dtype.itemsize * sta_loc.shape[1])))
        _, idx = np.unique(b, return_index=True)
        duplicate_list = list(set([i for i in range(dim)]) - set(idx))
        if len(idx) + len(duplicate_list) != dim:
            raise ValueError("The sum of dim doesn't agree")
        duplicate_list.sort()
        return duplicate_list

    def _sort_stations(self):
        sta_dict = dict()
        sta_list = list()
        for station in self.stations:
            sta_dict[station.tag] = station
        od = collections.OrderedDict(sorted(sta_dict.items()))
        for key, value in od.items():
            sta_list.append(value)
        self.stations = sta_list

    def _remove_duplicate_stations(self, duplicate_list):
        if len(duplicate_list) == 0:
            return
        for index in sorted(duplicate_list, reverse=True):
            del self.stations[index]
        if len(self._find_duplicate()) != 0:
            raise ValueError("There are still duplicates after removing.")

    def _sort_and_check_stations(self):
        self._sort_stations()
        duplicate_list = self._find_duplicate()
        self._remove_duplicate_stations(duplicate_list)
        print("Number of original stations: %d" % len(self._stations))
        print("Number of duplicate stations removed: %d" % len(duplicate_list))
        print("Number of remaining stations: %d" % len(self.stations))
        
        sta_tag = []
        sta_loc = []
        for station in self.stations:
            sta_tag.append(station.tag)
            sta_loc.append([station.latitude, station.longitude])
        sta_loc = np.array(sta_loc)

        nw_dict = dict()
        for station in self.stations:
            if station.network in nw_dict.keys():
                nw_dict[station.network] += 1
            else:
                nw_dict[station.network] = 1
        nw_dict = collections.OrderedDict(sorted(nw_dict.items()))

        return sta_tag, sta_loc, nw_dict

    @staticmethod
    def _normalize(array):
        return array / array.max()

    @staticmethod
    def _plot_matrix(matrix, title="", figname=None):

        fig = plt.figure(figsize=(12, 6.4))

        ax = fig.add_subplot(111)
        ax.set_title('colorMap: %s' % title)
        plt.imshow(matrix, interpolation='none')
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')

        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)

    @staticmethod
    def _plot_histogram(weights, title="", figname=None):
        fig = plt.figure()

        plt.hist(weights, bins=30, alpha=0.75)

        plt.xlabel("weight")
        plt.ylabel("Count")
        plt.title("Histrogram: %s" % title)
        plt.grid(True)
        
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)

    def write_weight(self, filename="weight.txt", order="station_tag"):
        order = order.lower()
        weight_dict = dict()
        if order not in ["station_tag", "weight"]:
            raise ValueError("Order must be 'station_tag' or 'weight'")
        for tag, weight in zip(self.station_tag, self.weight):
           weight_dict[tag] = weight
        if order == "station_tag":
            _sorted = sorted(weight_dict.items(), key=operator.itemgetter(0))
        elif order == "weight":
            _sorted = sorted(weight_dict.items(), key=operator.itemgetter(1))
        else:
            raise NotImplementedError

        with open(filename, 'w') as fh:
            for tag, weight in _sorted:
                fh.write("%-10s %15.5e\n" % (tag, weight))

    def plot_global_map(self, outputfile=None):
        """
        Plot global map of event and stations
        """
        # ax = plt.subplot(211)
        fig = plt.figure()
        ax = plt.gca()
        plt.title("Station and Event distribution")

        m = Basemap(projection='moll', lon_0=0.0, lat_0=0.0,
                    resolution='c')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 60.))
        m.drawmapboundary()

        cm = plt.cm.get_cmap('RdYlBu')
        x, y = m(self.station_loc[:, 1], self.station_loc[:, 0])
        m.scatter(x, y, 30, color=self.weight, marker="^", edgecolor="k",
                  linewidth='0.3', zorder=3, cmap=cm)
        plt.colorbar(shrink=0.8)

        cmt_lat = self.event.latitude
        cmt_lon = self.event.longitude
        src_x, src_y = m(cmt_lon, cmt_lat)
        m.scatter(src_x, src_y, 60, color="g", marker="o", edgecolor="k",
                  linewidth='0.3', zorder=3)

        if outputfile is None:
            plt.show()
        else:
            plt.savefig(outputfile)
            plt.close(fig)


class ExpWeight(WeightBase):
    
    def __init__(self, stations, event, l0=1.0, phi0=10.0, threshold=0.01):
        WeightBase.__init__(self, stations, event)

        self.l0 = l0
        self.phi0 = phi0
        self.threshold = threshold

        self.azi_weight_matrix = np.zeros([self.nstations, self.nstations])
        self.azi_weight = np.zeros(self.nstations)
        self.dist_weight_matrix = np.zeros([self.nstations, self.nstations])
        self.dist_weight = np.zeros(self.nstations)

    @staticmethod
    def _distance(lat1, lon1, lat2, lon2):
        """
        The distance(unit:degree) between 2 points on
        the sphere
        """
        return locations2degrees(lat1, lon1, lat2, lon2)

    @staticmethod
    def _azimuth(lat1, lon1, lat2, lon2):
        """
        The azimuth(unit:degree) starting from point1 to
        point 2
        """
        _, azi, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        return azi

    @staticmethod
    def _azimuth_matrix(azis):
        dim = azis.shape[0]
        azi_m = np.zeros([dim, dim])
        for _i in range(dim):
            for _j in range(_i+1, dim):
                azi_m[_i, _j] = azis[_i] - azis[_j]
                azi_m[_j, _i] = - azi_m[_i, _j]
        return azi_m

    def calculate_azimuth_weight(self):

        azis = np.zeros(self.nstations)
        for _i in range(self.nstations):
            sta_loc = self.station_loc[_i]
            azis[_i] = self._azimuth(sta_loc[0], sta_loc[1],
                                     self.event.latitude, self.event.longitude)
        
        azi_m = self._azimuth_matrix(azis)
        exp_azi_m = np.exp(-(azi_m / self.phi0)**2)
        # reset diagonal term to 0
        np.fill_diagonal(exp_azi_m, 0.0)
        self.azi_weight_matrix = exp_azi_m

        self.azi_weight = np.sum(exp_azi_m, axis=1)
        self.azi_weight = self._normalize(self.azi_weight)

    def _distance_matrix(self, locations):
        """
        calculate distance matrix
        """
        dim = locations.shape[0]
        dist_m = np.zeros([dim, dim])
        # calculate the upper part
        for _i in range(dim):
            for _j in range(_i+1, dim):
                loc_i = locations[_i]
                loc_j = locations[_j]
                dist_m[_i, _j] = \
                    self._distance(loc_i[0], loc_i[1],
                                   loc_j[0], loc_j[1])
                # symetric
                dist_m[_j, _i] = dist_m[_i, _j]
        return dist_m

    def calculate_distance_weight(self):
        dist_m = self._distance_matrix(self.station_loc)
        exp_dist_m = np.exp(-(dist_m / self.l0)**2)
        # reset diagonal term to 0
        np.fill_diagonal(exp_dist_m, 0.0)
        self.dist_weight_matrix = exp_dist_m 

        self.dist_weight = np.sum(exp_dist_m, axis=1)
        self.dist_weight = self._normalize(self.dist_weight)

    def calculate_weight(self):
        self.calculate_azimuth_weight()
        self.calculate_distance_weight()

        for _i in range(self.nstations):
            #comb_value = self.azi_weight[_i] * self.dist_weight[_i]
            comb_value = self.dist_weight[_i]
            #if comb_value < self.threshold:
            #    comb_value2 = self.threshold
            #else:
            #    comb_value2 = comb_value
            self.weight[_i] = np.log(1./comb_value)
            #print("Station, v1, w: %10s, %10.5f %10.5f" 
            #        % (self.stations[_i].tag, comb_value,
            #           self.weight[_i]))

        self.weight = self._normalize(self.weight)
        for _i in range(self.nstations):
            if self.weight[_i] < self.threshold:
                self.weight[_i] = self.threshold

    def plot_weight_matrix(self, outputdir=None, figformat="png"):
        if outputdir is None:
            figname_dist = None
            figname_azi = None
        else:
            figname_dist = os.path.join(outputdir, "distance_matrix.%s" 
                                        % figformat)
            figname_azi = os.path.join(outputdir, "azimuth_matrix.%s" 
                                       % figformat)

        self._plot_matrix(self.dist_weight_matrix, title="Distance",
                          figname=figname_dist)
        self._plot_matrix(self.azi_weight_matrix, title="Azimuth",
                          figname=figname_azi)

    def plot_weight_histogram(self, outputdir=None, figformat="png"):
        if outputdir is None:
            figname_dist = None
            figname_azi = None
            figname_all = None
        else:
            figname_dist = os.path.join(outputdir, "distance_weight_hist.%s" 
                                        % figformat)
            figname_azi = os.path.join(outputdir, "azimuth_weight_hist.%s" 
                                       % figformat)
            figname_all = os.path.join(outputdir, "combined_weight_hist.%s" 
                                       % figformat)

        self._plot_histogram(self.dist_weight, title="Distance",
                             figname=figname_dist)
        self._plot_histogram(self.azi_weight, title="Azimuth",
                             figname=figname_azi)
        self._plot_histogram(self.weight, title="Combined Weight",
                             figname=figname_all)


class VoronoiWeight(WeightBase):

    def __init__(self, stations, event, order=1.0):
        WeightBase.__init__(self, stations, event)

        self.points = None
        self.sv = None
        self.order = order

    def _transfer_coordinate(self, radius, center):
        """
        Transfer (longitude, latitude) to (x, y, z) on sphere(radius, center)
        """
        sphere_loc = np.zeros([self.nstations, 3])
        for _i in range(self.station_loc.shape[0]):
            lat = np.deg2rad(self.station_loc[_i, 0])
            lon = np.deg2rad(self.station_loc[_i, 1])
            sphere_loc[_i, 0] = radius * cos(lat) * cos(lon) + center[0]
            sphere_loc[_i, 1] = radius * cos(lat) * sin(lon) + center[1]
            sphere_loc[_i, 2] = radius * sin(lat) + center[2]
        return sphere_loc

    def calculate_weight(self):
        radius = 1.0
        center = np.array([0, 0, 0])
        self.points = self._transfer_coordinate(radius, center)
        #for _i in range(self.nstations):
        #    print("%10s: [%10.5f, %10.5f] -- [%10.5f, %10.5f, %10.5f]" 
        #          % (self.station_tag[_i], self.station_loc[_i][0], 
        #             self.station_loc[_i][1], self.points[_i][0], 
        #             self.points[_i][1], self.points[_i][2])) 
        
        self.sv = SphericalVoronoi(self.points, 1, center)
        self.sv.sort_vertices_of_regions()

        surface_area = self.sv.compute_surface_area()
        self.weight = surface_area ** self.order
        self.weight = self._normalize(self.weight)

    def plot_sphere(self):
        points = self.points
        sv = self.sv
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # plot the unit sphere for reference (optional)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v)) 
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='y', alpha=0.05)
        # plot Voronoi vertices
        #ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], 
        #           c='g') 
        # indicate Voronoi regions (as Euclidean polygons) 
        for region in sv.regions: 
            random_color = colors.rgb2hex(np.random.rand(3)) 
            polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0) 
            polygon.set_color(random_color) 
            ax.add_collection3d(polygon) 

        # plot generator points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1);
        ax.set_xticks([-1,1])
        ax.set_yticks([-1,1])
        ax.set_zticks([-1,1]);
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()

    def plot_weight_histogram(self, outputdir=None, figformat="png"):
        if outputdir is None:
            figname = None
        else:
            figname = os.path.join(outputdir, "voronoi_weight_hist.%s" 
                                       % figformat)

        self._plot_histogram(self.weight, title="Voronoi Weight",
                             figname=figname)


