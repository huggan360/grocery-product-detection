"""Queue the annotation-tool video library again, preserving old reviews/results."""
import argparse
from collections import Counter
import getpass
import json
import time

import requests


def main():
    p=argparse.ArgumentParser(description='Back up existing predictions/reviews and reclassify videos with the running reviewer models.')
    p.add_argument('--all',action='store_true',required=True,help='Explicitly queue every inactive video in the library.')
    p.add_argument('--url',default='http://127.0.0.1:9000')
    p.add_argument('--editor',default=getpass.getuser())
    p.add_argument('--wait',action='store_true',help='Print progress until this batch completes.')
    args=p.parse_args()
    url=args.url.rstrip('/')
    session=requests.Session()
    response=session.post(url+'/api/editors',json={'name':args.editor},timeout=30)
    response.raise_for_status()
    session.headers['X-Session']=response.json()['token']
    response=session.post(url+'/api/videos/reclassify',json={'all_videos':True},timeout=180)
    response.raise_for_status()
    batch=response.json()
    print(json.dumps(batch,indent=2),flush=True)
    if args.wait:
        ids=set(batch['video_ids'])
        while ids:
            response=session.get(url+'/api/videos',timeout=30);response.raise_for_status()
            counts=Counter(r['status'] for r in response.json()['items'] if r['id'] in ids)
            print(json.dumps(dict(counts)),flush=True)
            if counts.get('done',0)+counts.get('failed',0)==len(ids):
                if counts.get('failed',0):raise SystemExit('Some clips failed; inspect their messages in the annotation tool.')
                return
            time.sleep(5)


if __name__=='__main__':main()
