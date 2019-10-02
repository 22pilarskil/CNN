import os
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import argparse

flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
SCOPES = "https://www.googleapis.com/auth/drive.file"
store = file.Storage('storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets("/Users/michaelpilarski/Desktop/Liamsappclient_secret.json", scope=SCOPES)
    creds = tools.run_flow(flow, store, flags) \
    if flags else tools.run(flow, store)
http=creds.authorize(Http())
print("done")
#http = creds.refresh(http)
DRIVE = build('drive', 'v3', http)
MIMETYPE="text/plain"
ID = "1g6QRtm-X-Uzgw2IkWrZogtjrENzSYg3G2L2Cc24vqlk"
data=DRIVE.files().export(fileId=ID, mimeType=MIMETYPE).execute()
print(data)
#https://docs.google.com/document/d/1g6QRtm-X-Uzgw2IkWrZogtjrENzSYg3G2L2Cc24vqlk/edit?usp=sharing
#https://drive.google.com/open?id=1_lsXhzX8-STLLiyFsiQ0di4YXy8lpo5uW3dHXzAa7cI
