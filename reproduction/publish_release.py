"""Publish replay archives to the existing GitHub repository's release."""
import argparse
import os
import http.client
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import hashlib
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--credential-file', type=Path, required=True)
    ap.add_argument('--repository', default=os.environ.get('SORD_RELEASE_REPOSITORY', 'KheirAli/SORD'))
    ap.add_argument('--tag', default='reproduction-v1')
    ap.add_argument('--assets', type=Path, default=ROOT/'release-assets')
    ap.add_argument('--watch', action='store_true', help='upload completed archive parts while the builder is running')
    ap.add_argument('--extract-to', type=Path, help='extract uploaded parts into an independent validation checkout')
    ap.add_argument('--consume', action='store_true', help='remove generated local archive parts after upload and extraction')
    a = ap.parse_args()
    token = a.credential_file.read_text().strip()
    headers = {'Authorization':'Bearer '+token, 'Accept':'application/vnd.github+json', 'User-Agent':'SORD-reproduction', 'Content-Type':'application/json'}
    base = 'https://api.github.com/repos/'+a.repository
    def api(path, body=None, method=None):
        req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(), headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=60) as response:return json.load(response)
    releases = api('/releases')
    release = next((r for r in releases if r['tag_name']==a.tag),None)
    if release is None:
        release = api('/releases', {'tag_name':a.tag,'target_commitish':'main','name':'SORD saved-result reproduction assets','draft':True,'body':'Saved benchmark observations, clean targets, masks and reconstruction arrays for the registered CPU replay commands. NPZ archives retain only arrays read by those commands. This verifies metric replay, not fresh diffusion generation or retraining. See reproduction/PROVENANCE.md and assets.json for protocol differences and checksums.'})
    existing={p['name']:p for p in api('/releases/'+str(release['id'])+'/assets')}
    manifest_path=ROOT/'reproduction/assets.json'
    uploaded={}
    def upload(record):
        p=a.assets/record['name']
        if p.name in existing:
            if existing[p.name]['size']!=record['bytes']:raise RuntimeError('Existing release asset differs: '+p.name)
            result=existing[p.name]
        else:
            url=urllib.parse.urlparse(release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(p.name))
            connection=http.client.HTTPSConnection(url.hostname,timeout=300)
            upload_headers={**headers,'Content-Type':'application/x-tar','Content-Length':str(p.stat().st_size)}
            print('Uploading',p.name,p.stat().st_size,flush=True)
            with p.open('rb') as stream:
                connection.request('POST',url.path+'?'+url.query,body=stream,headers=upload_headers)
                response=connection.getresponse();data=response.read()
            if response.status not in (200,201):raise RuntimeError(f'Upload failed for {p.name}: HTTP {response.status}; '+data.decode()[:200])
            result=json.loads(data)
        record['url']=result['browser_download_url'];record['asset_id']=result['id']
        uploaded[p.name]=record
        (a.assets/'upload-state.json').write_text(json.dumps(uploaded,indent=2)+'\n')
        if a.extract_to:
            a.extract_to.mkdir(parents=True,exist_ok=True)
            with tarfile.open(p) as archive:archive.extractall(a.extract_to,filter='data')
        if a.consume:p.unlink()
        print('Uploaded',p.name,flush=True)
    if a.watch:
        index=1
        while True:
            p=a.assets/f'replay-{index:02d}.tar'
            following=a.assets/f'replay-{index+1:02d}.tar'
            manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            ready=following.exists() or (manifest is not None and any(r['name']==p.name for r in manifest['archives']))
            if p.exists() and ready:
                with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                upload({'name':p.name,'bytes':p.stat().st_size,'sha256':digest});index+=1
            elif manifest is not None and index>len(manifest['archives']):break
            else:time.sleep(5)
        manifest=json.loads(manifest_path.read_text())
        for record in manifest['archives']:
            actual=uploaded[record['name']]
            if actual['sha256']!=record['sha256']:raise RuntimeError('Builder/uploader checksum mismatch: '+record['name'])
            record.update(actual)
    else:
        manifest=json.loads(manifest_path.read_text())
        for record in manifest['archives']:upload(record)
    published=api('/releases/'+str(release['id']), {'draft':False},method='PATCH')
    # GitHub draft assets have temporary untagged download URLs; refresh
    # them after publishing the release.
    final_assets={p['name']:p for p in api('/releases/'+str(release['id'])+'/assets')}
    for record in manifest['archives']:
        record['url']=final_assets[record['name']]['browser_download_url']
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if a.extract_to:
        target=a.extract_to/'reproduction/assets.json';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(manifest_path,target)
    print('Published',published['html_url'],flush=True)

if __name__=='__main__':main()
