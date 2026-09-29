"""Bounded downloads and ZIP inventory checks for checksum-pinned research sources."""
import hashlib
from pathlib import PurePosixPath
import urllib.request


def fetch_archive(path,url,size,md5):
    if not 0<size<=500_000_000:
        raise ValueError('Archive exceeds per-source download budget')
    def validate(file):
        digest=hashlib.md5()
        with file.open('rb') as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
        if file.stat().st_size!=size or digest.hexdigest()!=md5:
            raise ValueError('Publisher archive size or checksum mismatch')
    if path.exists():validate(path);return path
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part')
    if partial.exists():raise ValueError('Interrupted partial archive exists; inspect before restarting')
    request=urllib.request.Request(url,headers={'User-Agent':'DXA-QC-research'})
    with urllib.request.urlopen(request,timeout=45) as response,partial.open('xb') as stream:
        count=0
        while chunk:=response.read(1024*1024):
            count+=len(chunk)
            if count>size:raise ValueError('Download exceeds pinned byte budget')
            stream.write(chunk)
            if count%(64*1024*1024)==0:print(f'Downloaded {count//(1024*1024)} MiB',flush=True)
    validate(partial);partial.rename(path)
    return path


def safe_members(archive,count_budget=20000,byte_budget=2_000_000_000):
    entries=archive.infolist()
    if len(entries)>count_budget or sum(i.file_size for i in entries)>byte_budget:
        raise ValueError('ZIP extraction budget exceeded')
    seen=set()
    for item in entries:
        path=PurePosixPath(item.filename)
        if (path.is_absolute() or '..' in path.parts or '\\' in item.filename or
            ((item.external_attr>>16)&0o170000)==0o120000 or item.filename in seen):
            raise ValueError('Unsafe or duplicated ZIP member')
        seen.add(item.filename)
    return entries
