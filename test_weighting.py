from __future__ import print_function
from spaceweight import ExpWeight, VoronoiWeight, Station, Event
from pprint import pprint
import collections
import matplotlib.pyplot as plt


def read_stations(station_file):
    stations = []
    with open(station_file) as fh:
        content = fh.readlines()
    
    for line in content:
        info = line.split()
        sta = info[0]
        nw = info[1]
        lat = float(info[2])
        lon = float(info[3])
        elev = float(info[4])
        depth = float(info[5])
        stations.append(Station(network=nw, station=sta, latitude=lat,
                                longitude=lon, elevation=elev))
    
    return stations


def test_exp(stations, event):
    weight = ExpWeight(stations, event, l0=20.0, phi0=10.0,
                       threshold=0.01)
    weight.calculate_weight()

    #azi_weight = weight.azi_weight
    #print("azi cond-number:", max(azi_weight)/min(azi_weight))

    #dist_weight = weight.dist_weight
    #print("dist cond-number:", max(dist_weight)/min(dist_weight))

    #weight.plot_weight_matrix()
    #weight.plot_weight_histogram()
    weight.write_weight(filename="exp_weight.txt", order="weight")
    #weight.plot_global_map()
    return weight.weight

def test_vor(stations, event):
    weight = VoronoiWeight(stations, event)
    weight.calculate_weight()

    #weight.plot_global_map()
    #weight.plot_sphere()

    #weight.plot_weight_histogram()
    weight.write_weight(filename="vor_weight.txt")

    return weight.weight

if __name__ == "__main__":
    stations = read_stations("./examples/STATIONS")
    event = Event(latitude=0.0, longitude=0.0)

    expdata = test_exp(stations, event)
    vordata = test_vor(stations, event)
    
    print(min(expdata), max(expdata))
    #print(expdata)
    #print(vordata)

    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    plt.plot(expdata, vordata, 'b*')
    plt.plot((0., 1.), (0., 1.), 'k--')
    plt.axis('equal')

    plt.xlabel("Exponential Weight")
    plt.ylabel("Voronoi Weight")
    #plt.xlim(0.,1.)
    #plt.ylim(0.,1.)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    plt.grid()
    plt.show()

    with open("ratio.txt", "w") as fh:
        for _i in range(len(stations)):
            ratio = vordata[_i]/expdata[_i]*100.
            fh.write("%10.5f %10.5f %10.2f\n" % (expdata[_i], vordata[_i], 
                     ratio))


