from __future__ import print_function
import httplib2
import os

from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = 'https://www.googleapis.com/auth/drive'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Drive API Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'drive-quickstart.json')

    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatability with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def get_service():
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('drive', 'v2', http=http)
    return service

def files():
    """Get all files in My Drive

    Creates a Google Drive API service object and returns file
    dictionaries for all files.

    """
    service = get_service()

    results = service.files().list(maxResults=1000).execute()
    return results


class Node:
    def __init__(self, info):
        self.info = info
        self.parents = []
        self.children = []

    def __str__(self):
        return '{}: {}'.format(self.type, self.title)

    def __repr__(self):
        return str(self)

    @property
    def type(self):
        return 'Folder' if self.is_folder else 'File'

    @property
    def id(self):
        return self.info['id']
    
    @property
    def title(self):
        return self.info['title']

    folder_mimetype = 'application/vnd.google-apps.folder'
    @property
    def is_folder(self):
        return self.info['mimeType'] == Node.folder_mimetype
    
    def has_parent(self, parent):
        if self.parents:
            if any(p == parent for p in self.parents):
                return True
            if any(p.has_parent(parent) for p in self.parents):
                return True
        return False
        
def get_file_graph():
    R = files()
    nodes = {'root': Node({'id': '0','title': 'root',
                           'mimeType': Node.folder_mimetype})}
    root = nodes['root']
    for r in R['items']:
        n = Node(r)
        nodes[r['id']] = n
    for r in R['items']:
        n = nodes[r['id']]
        if 'parents' in r:
            for p in r['parents']:
                if p['isRoot']:
                    n.parents.append(root)
                    root.children.append(n)
                else:
                    # if p['id'] not in nodes, then it is a folder not
                    # owned by this user (and probably not relevant)
                    if p['id'] in nodes:
                        parent = nodes[p['id']]
                        n.parents.append(parent)
                        parent.children.append(n)
    return root

def get_node(root, path):
    if path == '/':
        parts = []
    else:
        if path[0] == '/':
            path = path[1:]
        parts = path.split('/')

    node = root
    for p in parts:
        try:
            node = [c for c in node.children if c.title == p][0]
        except IndexError:
            raise IndexError("No node with that path")
    return node

def walk(root, path):
    node = get_node(root, path)
    if not node.is_folder:
        yield node.parents[0],[],[node]
        raise StopIteration
    stack = [node]
    while stack:
        node = stack.pop()
        files = [c for c in node.children if not c.is_folder]
        dirs = [c for c in node.children if c.is_folder]
        yield node, dirs, files

        stack.extend(dirs)



if __name__ == '__main__':
    main()
