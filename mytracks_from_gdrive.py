import os

import gdrive
import mytracks

def get_gdrive_mytracks():
    root = gdrive.get_file_graph()
    kml = []
    for rnode, dirs, files in gdrive.walk(root,'/Shared Data'):
        kml.extend(f for f in files if (f.title.endswith('.kmz') or
                                        f.title.endswith('.kmz')))
    
    service = gdrive.get_service()
    for fnode in kml:
        data = service.files().get_media(fileId=fnode.id).execute()
        print("Writing out:", fnode.title)
        with open(fnode.title,'wb') as wfp:
            wfp.write(data)


def main():
    get_gdrive_mytracks()

if __name__=='__main__':
    main()
