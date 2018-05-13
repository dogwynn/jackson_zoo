#!/usr/bin/env python3
import logging
import re
import os
import collections
import html
import argparse
import zipfile

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager

import fastkml.kml as kml
import fastkml.styles as kml_styles
from shapely.geometry import (
    Polygon, Point
)
from PIL import Image, ImageDraw, ImageFont

from memoized_property import memoized_property

import mytracks

DEFAULT_ALPHA = 1

def int_to_hex(i):
    return hex(i).split('x')[-1].zfill(2)

float_re_s = r'(?<![\w/])[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?![\w/])'
float_re = re.compile(r'({})'.format(float_re_s))
# key_re_s = r'\S+(?:\s\d)?'
key_re_s = r'.*?'
key_re = re.compile(r'({}){}'.format(key_re_s,float_re_s))
# key_only_re = re.compile('{}\s'.format(key_re_s))
# Key_re = re.compile(r'(.*?){}'.format(float_re_s))

def parse_marker_name(name):
    key = key_re.findall(name)[0].strip() if key_re.search(name) else ''
    floats = [float(v) for v in float_re.findall(name)]
    return key, floats

def test_parse_marker_name():
    assert parse_marker_name('test 81 82 83') == ('test',[81,82,83])
def test_parse_marker_name2():
    assert parse_marker_name('test -81 82 83') == ('test',[-81,82,83])
def test_parse_marker_name3():
    assert parse_marker_name('test ,-81 82 83') == ('test ,',[-81,82,83])
    
def get_temp_data(markers):
    temps = collections.OrderedDict()
    for marker_set in markers:
        for name, ts, pos in marker_set['markers']:
            key, floats = parse_marker_name(name)
            if floats:
                if floats[0] < 50:
                    floats = [f*9/5+32 for f in floats]
                df = pd.DataFrame(floats,columns=['temp_f'],)
                temps[key] = (ts, pos, df)
    return temps

def get_bbox(P,scale=1.0):
    minx, miny = np.min(P,axis=0)
    maxx, maxy = np.max(P,axis=0)
    dx = (maxx - minx); dx2 = dx/2
    dy = (maxy - miny); dy2 = dy/2
    cx,cy = minx+dx2, miny+dy2
    dx2 *= scale
    dy2 *= scale
    minx = cx - dx2
    miny = cy - dy2
    maxx = cx + dx2
    maxy = cy + dy2
    return np.array([(minx, miny), (maxx, miny),
                     (maxx, maxy), (minx, maxy)])

def test_get_bbox():
    assert np.array_equal(get_bbox(np.array([[0,0],[1,1]])),
                          np.array([[0,0],[1,0],[1,1],[0,1]]))

    assert np.array_equal(get_bbox(np.array([[0,0],[1,1]]), scale=2),
                          np.array([[-0.5,-0.5],[1.5,-0.5],
                                    [1.5,1.5],[-0.5,1.5]]))

def perturb_overlap(positions):
    seen = set()
    new = []
    bbox = get_bbox()
    #positions p in positions

def poly_area(P):
    X,Y = map(lambda a:a.reshape(len(a)), np.hsplit(P,2))
    return 0.5*np.abs(np.dot(X,np.roll(Y,1)) - np.dot(Y,np.roll(X,1)))

def get_finite_regions(V):
    return [(i,r) for i,r in enumerate(V.regions) if -1 not in r]

def get_areas(V):
    areas = []
    for ri,vertices in get_finite_regions(V):
        P = np.array([V.vertices[i] for i in vertices])
        area = poly_area(P)
        areas.append(area)
    return np.array(areas)

def get_avg_distance(V):
    dvs = np.array([np.linalg.norm(V.points[i+1]-V.points[i])
                    for i in range(len(V.points)-1)])
    return dvs
    return dvs.mean()

def region_points(V):
    lut = {r:[] for r in range(len(V.regions))}
    # don't count bbox input points
    for i,r in enumerate(V.point_region[:-4]):
        lut[r].append(i)
    return lut


def get_voronoi_polys(V):
    return [(i,[list(V.vertices[v]) for v in region])
            for i,region in get_finite_regions(V)]

def normalize_voronoi_polys(P):
    mm_lat, mm_long = (
        (min([p[0] for _, points in P for p in points]),
         max([p[0] for _, points in P for p in points])),
        (min([p[1] for _, points in P for p in points]),
         max([p[1] for _, points in P for p in points])),
    )
    print(mm_lat)
    print(mm_long)
    return [
        (ri,
         [((p[0] - mm_lat[0])/(mm_lat[1]-mm_lat[0]),
           (p[1] - mm_long[0])/(mm_long[1]-mm_long[0]))
          for p in points])
        for ri, points in P
    ]

class TracksAnalysis:
    '''Temperature analysis and geo-tagging for MyTracks kmz files

    Given a list of MyTracks (.kmz/.kml) file paths:
    
    1) Parse out temperatures, positions, timestamps from the
       waypoints
    2) Save positions for each waypoint

    This will be used to produce circle-bounded Voronoi diagrams,
    colored by relative temperature, and overlayed on a geo-tagged (KML)
    output file.

    Args:

    *paths: paths to .kmz/.kml files to be parsed as MyTracks files

    '''
    def __init__(self, *paths):
        tracks = [mytracks.MyTracks(p) for p in paths]

        # tracks sets for each MyTracks object
        self.tracks = [list(t.tracks()) for t in tracks]

        # waypoint marker sets for each MyTracks object
        self.markers = [list(t.markers()) for t in tracks]

        # temperature data for each MyTracks waypoint marker set
        self.temp_data = [get_temp_data(m) for m in self.markers]

        # (lat,long) position for each waypoint in each MyTracks
        # object
        try:
            self.latlongs = [np.array([pos[:2] if pos else None for _,pos,_ in tdata.values()])
                              for tdata in self.temp_data]
        except:
            raise

    def delta_overlapping_positions(self):
        # TODO: need to handle case where multiple waypoints have same
        # (lat,long)
        pass


class WaypointVoronoi(Voronoi):
    @classmethod
    def from_positions(cls, positions):
        # get a bounding box for the points to find a good 
        bbox = get_bbox(positions)

        bbox = get_bbox(positions, scale=4)

        V = cls(
            np.concatenate((positions,bbox),axis=0),
            incremental=True,
        )
        return V

    def init_from_track_data(self, latlongs, temp_data):
        # Mapping from Voronoi point index to (track, waypoint) index
        # pair
        index_map = {}
        index = 0
        for i in range(len(latlongs)):
            for j in range(len(latlongs[i])):
                index_map[index] = (i,j)
                index += 1

        # Build region index => temp data map
        region_data = {}
        r2p = region_points(self)
        for ri in range(len(self.regions)):
            for pi in r2p[ri]:
                track,waypoint = index_map[pi]
                data_list = region_data.setdefault(ri,[])
                track_data = list(temp_data[track].items())
                key, (ts, pos, df) = track_data[waypoint]
                data_list.append((key, ts, pos, df))
        
        region_dfs, region_names = {}, {}
        for ri, temps in region_data.items():
            keys, dfs = [], []
            for (key, ts, pos, df) in temps:
                keys.append(key)
                dfs.append(df)
            region_names[ri] = keys
            region_dfs[ri] = dfs

        self.region_dfs = region_dfs
        self.region_dfs_concat = {ri: pd.concat(dfs)
                                  for ri,dfs in region_dfs.items()}
        self.region_names = region_names

        region_descriptions = {}
        for ri, names in region_names.items():
            dfs = self.region_dfs_concat[ri].sort_values(['temp_f'])
            avg_temp = dfs.temp_f.mean()
            dfs.columns = ['Temp (F)']
            desc = '<p>{name}</p><p>Average temp: {avg}</p><p>{temps}</p>'.format(
                name=self.region_name(ri),
                avg=avg_temp,
                temps=dfs.to_html(index=False),
            )
            region_descriptions[ri] = desc
        self.region_descriptions = region_descriptions

    @classmethod
    def from_tracks(cls, T):
        positions = np.concatenate(T.latlongs,axis=0)
        V = cls.from_positions(positions)
        V.init_from_track_data(T.latlongs, T.temp_data)
        return V

    @classmethod
    def from_tracks_normalized(cls, T, width=1, height=1):
        positions = np.concatenate(T.latlongs,axis=0)
        mm = positions.min(axis=0), positions.max(axis=0)
        def norm(ll):
            return (((ll - mm[0]) / (mm[1] - mm[0])) *
                    np.array([width, height]))
        latlongs = [norm(ll) for ll in T.latlongs]
        normed = norm(positions)
        # print(latlongs)
        V = cls.from_positions(normed)
        V.init_from_track_data(latlongs, T.temp_data)
        return V

    def region_avg_temps(self):
        rdfs = sorted(self.region_dfs_concat.items())
        region_avgs = [(ri,dfs.temp_f.mean()) for ri,dfs in rdfs]
        return region_avgs

    def init_color(self, colormap=cm.RdYlBu_r, alpha=DEFAULT_ALPHA ):
        rdfs = sorted(self.region_dfs_concat.items())
        region_avgs = [(ri,dfs.temp_f.mean()) for ri,dfs in rdfs]
        avgs = [avgt if avgt<100 else 120 for _,avgt in region_avgs]
        mn,mx = min(avgs), min(max(avgs)+5,120)
        norm_avgs = [(v-mn)/(mx-mn) for v in avgs]

        colors = colormap(norm_avgs,alpha=alpha)
        
        self.region_to_color =  {rdfs[i][0]: colors[i]
                                 for i in range(len(colors))}

    region_to_color = {}
    def region_color(self, region_index, default=(0.5,0.5,0.5,0.5),
                     rev_hexed=True):
        color_array = self.region_to_color.get(region_index,default)
        colors = [int(255*v) for v in color_array]
        if rev_hexed:
            return '#{}'.format(''.join(map(int_to_hex,reversed(colors))))
        else:
            return tuple(colors)

    region_names = {}
    def region_name(self, region_index, default=[]):
        name_list = self.region_names.get(region_index,default)
        return ' | '.join(name_list)

    region_descriptions = {}
    def region_description(self, region_index, default=''):
        return self.region_descriptions.get(region_index, default)

    def to_image(self, alpha=DEFAULT_ALPHA, buf=0.5):
        self.init_color(alpha=alpha, colormap=cm.Greys)

        W, H = 900, 900
        base = Image.new('RGBA', (W, H))
        # base = Image.new('RGB', (W, H), color='white')
        draw = ImageDraw.Draw(base)
        arial_paths = [f for f in font_manager.findSystemFonts()
                       if 'arial.ttf' in f.lower()]
        font = ImageFont.truetype(arial_paths[0], 36)

        region_avgs = dict(self.region_avg_temps())
        
        r2p = region_points(self)
        areas = get_areas(self)
        avg_area = areas.mean(axis=0)
        area_std = areas.std(axis=0)
        vor_polys = get_voronoi_polys(self)
        intersects = []
        for i, (ri, P) in enumerate(vor_polys):
            region_poly = Polygon(P)
            center = Point(*self.points[r2p[ri][0]])
            # circle = center.buffer(np.sqrt(avg_area/np.pi))
            circle = center.buffer(buf*avg_area)
            intersect = circle.intersection(region_poly)
            intersects.append(intersect)

        positions = np.concatenate(
            [i.boundary.coords for i in intersects], axis=0
        )
        mm = positions.min(axis=0), positions.max(axis=0)
        def norm(P, width=W, height=H):
            return (((P - mm[0]) / (mm[1] - mm[0])) *
                    np.array([width, height]))

        hot_region = 1
        hot_regions = {}
        for i, (ri, P) in enumerate(vor_polys):
            intersect = intersects[i]
            name = self.region_name(ri)

            coords = np.array(intersect.boundary.coords)
            # print(norm(coords))
            draw.polygon([(c[0], H-c[1]) for c in norm(coords)],
                         fill=self.region_color(ri, rev_hexed=False),
                         outline=(255, 255, 255, 255))
            

            # style_name = 'region-{}'.format(ri)
            # style = kml.Style(ns, style_name)
            # rstyle = kml_styles.PolyStyle(ns, color=self.region_color(ri))
            # style.append_style(rstyle)
            # doc.append_style(style)

            # mark = kml.Placemark(
            #     ns, id=style_name, name=self.region_name(ri),
            #     description=self.region_description(ri),
            #     styleUrl='#{}'.format(style_name),
            # )
            # mark.geometry = intersect
            # folder.append(mark)

            if region_avgs[ri] > 100:
                center = intersect.centroid
                # center = Point(*self.points[r2p[ri][0]])
                print(norm(np.array(center.coords)))
                circle = center.buffer(0.02)
                coords = norm(np.array(circle.boundary.coords))
                c = coords[0] - np.array([20,-18])
                c[1] = H-c[1]
                draw.multiline_text(
                    (c[0], c[1]),
                    #name,
                    str(hot_region),
                    fill=(255, 255, 255, 255),
                    align='center',
                    font=font,
                )
                hot_regions[hot_region] = {
                    'name': name,
                }
                hot_region += 1
                # print(coords)
                # draw.polygon([(c[0], H-c[1]) for c in coords],
                #              fill=(255, 255, 255, 255))
                
            #     point_mark = kml.Placemark(
            #         ns, id=style_name, name=self.region_name(ri),
            #         description=self.region_description(ri),
            #         # styleUrl='#{}'.format(style_name),
            #     )
            #     point_mark.geometry = Point(*self.points[r2p[ri][0]])
            #     folder.append(point_mark)
        x, y = (50, 50)
        text = '\n'.join(
            f'{r}: {d["name"]}' for r,d in hot_regions.items()
        )
        draw.multiline_text((x, y), text, fill=(0, 0, 0, 255), font=font)

        return base
        
    def to_kml(self, alpha=DEFAULT_ALPHA):
        self.init_color(alpha=alpha)

        region_avgs = dict(self.region_avg_temps())
        
        ns = '{http://www.opengis.net/kml/2.2}'
        doc = kml.Document(ns, 'temp','Temps from MyTracks',
                           'Temps from MyTracks')
        folder = kml.Folder(ns, 'waypoints', 'waypoints', 'waypoints')
        doc.append(folder)

        r2p = region_points(self)
        areas = get_areas(self)
        avg_area = areas.mean(axis=0)
        area_std = areas.std(axis=0)
        for i, (ri, P) in enumerate(get_voronoi_polys(self)):
            region_poly = Polygon(P)
            center = Point(*self.points[r2p[ri][0]])
            # circle = center.buffer(np.sqrt(avg_area/np.pi))
            circle = center.buffer(avg_area*200)
            intersect = circle.intersection(region_poly)

            style_name = 'region-{}'.format(ri)
            style = kml.Style(ns, style_name)
            rstyle = kml_styles.PolyStyle(ns, color=self.region_color(ri))
            style.append_style(rstyle)
            doc.append_style(style)

            mark = kml.Placemark(
                ns, id=style_name, name=self.region_name(ri),
                description=self.region_description(ri),
                styleUrl='#{}'.format(style_name),
            )
            mark.geometry = intersect
            folder.append(mark)

            if region_avgs[ri] > 100 or True:
                point_mark = kml.Placemark(
                    ns, id=style_name, name=self.region_name(ri),
                    description=self.region_description(ri),
                    # styleUrl='#{}'.format(style_name),
                )
                point_mark.geometry = Point(*self.points[r2p[ri][0]])
                folder.append(point_mark)

        return doc

def output_kml(kml, path):
    with open(path, 'wt') as wfp:
        wfp.write(kml.to_string())

def output_kmz(kml, path):
    with zipfile.ZipFile(path, 'w') as wzf:
        wzf.writestr('doc.kml',kml.to_string())

def get_args():
    def exists(path):
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
        raise argparse.ArgumentTypeError(
            "Path {} does not exist.".format(path)
        )

    def valid_alpha(alpha):
        alpha = float(alpha)
        if alpha>1:
            alpha = alpha/255
        elif alpha < 0:
            raise argparse.ArgumentTypeError(
                "Invalid alpha ({})".format(alpha)
            )
        return alpha

    parser = argparse.ArgumentParser()
    
    parser.add_argument('paths',type=exists, nargs='+')
    parser.add_argument(
        '-a','--alpha', type=valid_alpha,
        default=DEFAULT_ALPHA,
    )
    parser.add_argument(
        '-k','--kml', action='store_true',
    )
    parser.add_argument(
        '-I','--image', action='store_true',
    )
    parser.add_argument(
        '-o','--output', type=os.path.abspath,
    )
    parser.add_argument('--gzip', action='store_true')

    return parser.parse_args()

def main():
    args = get_args()

    T = TracksAnalysis(*args.paths)
    V = WaypointVoronoi.from_tracks(T)

    if args.kml:
        kml = V.to_kml(alpha=args.alpha)
        if args.output:
            if args.gzip:
                output_kmz(kml, args.output)
            else:
                output_kml(kml, args.output)
        else:
            print(kml.to_string())
    if args.image:
        base = V.to_image(alpha=args.alpha)
        if not args.output:
            raise IOError('need output (-o) specified for png option')
        base.save(args.output)
        

if __name__=='__main__':
    main()
        
