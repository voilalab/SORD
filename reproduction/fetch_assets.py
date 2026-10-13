"""Download, verify and extract the GitHub release's CPU replay inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
        if redirected and urllib.parse.urlparse(req.full_url).hostname != urllib.parse.urlparse(newurl).hostname:
            redirected.remove_header('Authorization')
        return redirected

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',default=os.environ.get('SORD_RELEASE_REPOSITORY','ANONYMOUS/SORD'),
                    help='GitHub owner/name hosting the reproduction-v1 release (or set SORD_RELEASE_REPOSITORY)')
    ap.add_argument('--token-env',default='GH_TOKEN',help='environment variable holding a token for private release access')
    a=ap.parse_args()
    manifest=json.loads((ROOT/'reproduction/assets.json').read_text())
    dest=ROOT/'release-assets';dest.mkdir(exist_ok=True)
    token=os.environ.get(a.token_env)
    opener=urllib.request.build_opener(SafeRedirect())
    for record in manifest['archives']:
        p=dest/record['name']
        headers={'User-Agent':'SORD-reproduction','Accept':'application/octet-stream'}
        if token:
            headers['Authorization']='Bearer '+token
            url=f"https://api.github.com/repos/{a.repository}/releases/assets/{record['asset_id']}"
        else:url=record['url'].format(repository=a.repository)
        if not p.is_file():
            print('Downloading',p.name,flush=True)
            partial=p.with_suffix(p.suffix+'.part')
            with opener.open(urllib.request.Request(url,headers=headers),timeout=120) as response,partial.open('wb') as out:
                while data:=response.read(1024*1024):out.write(data)
            with partial.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=record['sha256']:raise RuntimeError('Downloaded archive checksum mismatch: '+p.name)
            partial.replace(p)
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=record['sha256']:raise RuntimeError('Archive checksum mismatch: '+p.name)
        with tarfile.open(p) as archive:
            for member in archive.getmembers():
                if not member.isfile() or not (ROOT/member.name).resolve().is_relative_to(ROOT):
                    raise RuntimeError('Unsafe archive entry: '+member.name)
            archive.extractall(ROOT,filter='data')
        print('Extracted',p.name,flush=True)
    from verify_assets import main as verify
    return verify()

if __name__=='__main__':raise SystemExit(main())
