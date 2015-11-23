import os
import sys
import zipfile
from xml.etree import ElementTree as etree
from functools import wraps
# from lxml.etree import ElementTree as etree

import dateutil.parser

def memoized_property(fget):
    """
    Return a property attribute for new-style classes that only calls its getter on the first
    access. The result is stored and on subsequent accesses is returned, preventing the need to
    call the getter any more.
    Example::
        >>> class C(object):
        ...     load_name_count = 0
        ...     @memoized_property
        ...     def name(self):
        ...         "name's docstring"
        ...         self.load_name_count += 1
        ...         return "the name"
        >>> c = C()
        >>> c.load_name_count
        0
        >>> c.name
        "the name"
        >>> c.load_name_count
        1
        >>> c.name
        "the name"
        >>> c.load_name_count
        1
    """
    attr_name = '_{0}'.format(fget.__name__)

    @wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)

    return property(fget_memoized)

class MyTracks:
    def __init__(self, path):
        self.path = path
        self.check_extension()
        self.set_ns()
    
    def check_extension(self):
        _, ext = os.path.splitext(self.path)
        if ext not in {'.kmz','.kml'}:
            raise IOError("Don't know how to open file with extension: {}".format(ext))

    def fp(self):
        _, ext = os.path.splitext(self.path)
        if ext == '.kmz':
            fp = zipfile.ZipFile(self.path).open('doc.kml','r')
        else:
            fp = open(path,'r')
        return fp

    def set_ns(self):
        ns = {}
        default_ns = None
        with self.fp() as fp:
            for event, elem in etree.iterparse(fp, ('start-ns',)):
                if event == 'start-ns':
                    key, uri = elem
                    if not key:
                        key = 'kml'
                    ns[key] = uri
        self.ns = ns

    @memoized_property
    def document(self):
        with self.fp() as fp:
            tree = etree.parse(fp)

        root = tree.getroot()
        return root.find('kml:Document',self.ns)
        

    def markers(self):
        D = self.document

        folder_path = 'Folder'
        for element in self.findall(D, folder_path):
            marker = {}
            name_elem = self.find(element, 'name')
            if name_elem:
                marker['name'] = name_elem.text
            
            marker_path = 'Placemark'
            names = [e.text for e in self.findall(element,marker_path+'/name')]
            time_stamps = [
                dateutil.parser.parse(e.text)
                for e in self.findall(element,marker_path+'/TimeStamp/when')
            ]
            coordinates = [
                tuple(map(float,e.text.split(',')))
                for e in self.findall(element,marker_path+'/Point/coordinates')
            ]
            marker['markers'] = list(zip(names, time_stamps, coordinates))

            yield marker

    def tracks(self):
        D = self.document 

        path = 'Placemark/gx:MultiTrack'
        for mt_element in self.findall(D, path):
            mtrack = {'name': '', 'tracks': []}

            name_elem = self.find(mt_element, 'name')
            if name_elem:
                mtrack['name'] = name_elem.text

            t_path = 'gx:Track'
            for t_elem in self.findall(mt_element,t_path):
                track = []
                ts_path = 'when'
                timestamps = [
                    dateutil.parser.parse(e.text)
                    for e in self.findall(t_elem,ts_path)
                ]

                coord_path = 'gx:coord'
                coords = [
                    tuple(map(float,e.text.split()))
                    for e in self.findall(t_elem,coord_path)
                ]

                xdata_path = 'ExtendedData/SchemaData'
                speed_path = (
                    xdata_path + "/gx:SimpleArrayData[@name='speed']/gx:value"
                )
                speeds = [float(e.text) for e in self.findall(D,speed_path)]

                bearing_path = (
                    xdata_path + "/gx:SimpleArrayData[@name='bearing']/gx:value"
                )
                bearings = [float(e.text) for e in self.findall(D,speed_path)]

                accuracy_path = (
                    xdata_path + "/gx:SimpleArrayData[@name='accuracy']/gx:value"
                )
                accuracies = [float(e.text) for e in self.findall(D,speed_path)]

                # ignore speed, etc. for now
                mtrack['tracks'].append(list(zip(timestamps, coords)))
            yield mtrack

    def find(self, E, path):
        path = '/'.join([p if ':' in p else 'kml:'+p for p in path.split('/')])
        return E.find(path,self.ns)
    def findall(self, E, path):
        path = '/'.join([p if ':' in p else 'kml:'+p for p in path.split('/')])
        return E.findall(path,self.ns)


