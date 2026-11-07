"""Extract SPEED+ once and create FastPose-ViT view using hard links."""
from pathlib import Path,PurePosixPath
import zipfile,json,os,shutil,stat,math,time
archive=Path('/data/lost+found/speedplusv2.zip')
root=Path('/data/datasets');raw=root/'speedplusv2';stage=root/'.speedplusv2_extract'
fixed=root/'SPEED_PLUS_FIXED';view=root/'.SPEED_PLUS_FIXED_build'
assert not raw.exists() and not fixed.exists() and not stage.exists() and not view.exists(), 'Destination already exists'
root.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 infos=z.infolist();needed=sum(((i.file_size+4095)//4096)*4096 for i in infos)
 available=shutil.disk_usage(root).free
 assert available>needed+1024**3, f'Insufficient space including reserve: {available} / {needed}'
 stage.mkdir();files=0;byte_count=0;tick=time.monotonic()
 for info in infos:
  rel=PurePosixPath(info.filename)
  assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='speedplusv2'
  assert not stat.S_ISLNK(info.external_attr>>16), 'Unexpected symlink'
  target=stage.joinpath(*rel.parts[1:]);assert target.resolve().is_relative_to(stage.resolve())
  if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with z.open(info) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
  assert target.stat().st_size==info.file_size
  files+=1;byte_count+=info.file_size
  if time.monotonic()-tick>15:
   print(f'EXTRACT files={files} bytes={byte_count}/{needed}',flush=True);tick=time.monotonic()
 stage.rename(raw)
print('Extraction and ZIP CRC checks complete',files,byte_count,flush=True)
view.mkdir();counts={};all_names=set();provenance={};boxes=json.loads(Path('/data/FastPose-ViT/assets/speed_plus_bbox_annotations.json').read_text())
for split,sources in [('train',[('synthetic','train.json','')]),('val',[('synthetic','validation.json','')]),('test',[('lightbox','test.json','_lightbox'),('sunlamp','test.json','_sunlamp')])]:
 target_dir=view/'images'/split;target_dir.mkdir(parents=True);rows=[]
 for domain,json_name,suffix in sources:
  data=json.loads((raw/domain/json_name).read_text())
  for item in data:
   filename=item['filename'];assert Path(filename).name==filename
   destname=Path(filename).stem+suffix+Path(filename).suffix
   assert destname not in all_names;all_names.add(destname)
   q=item['q_vbs2tango_true'];t=item['r_Vo2To_vbs_true']
   assert len(q)==4 and len(t)==3 and all(math.isfinite(x) for x in q+t)
   assert abs(sum(x*x for x in q)-1)<0.01 and t[2]>0
   b=boxes[destname];assert 0<=b['x1']<b['x2']<=1920 and 0<=b['y1']<b['y2']<=1200
   source=raw/domain/'images'/filename;assert source.is_file()
   os.link(source,target_dir/destname)
   row=dict(item);row['filename']=destname;rows.append(row)
   provenance[destname]=str(source.relative_to(raw))
 (view/(split+'.json')).write_text(json.dumps(rows,indent=2));counts[split]=len(rows)
 print('PREPARED',split,len(rows),flush=True)
assert len(all_names)==len(boxes)==69491
shutil.copy2('/data/FastPose-ViT/assets/speed_plus_bbox_annotations.json',view/'speed_plus_bbox_annotations.json')
for name in ['camera.json','LICENSE.md']:shutil.copy2(raw/name,view/name)
(view/'image_sources.json').write_text(json.dumps(provenance,indent=2))
report={'archive':str(archive),'archive_sha256':'062ad3d1309841fe072330c745790181cfcfede6fe3c7d4164003103ed7568d3','raw':str(raw),'dataset':str(fixed),'counts':counts,'test_domains':{'lightbox':6740,'sunlamp':2791},'bbox_count':len(boxes),'storage':'hard links; images occupy disk once','crc_verified':True,'uncompressed_bytes':byte_count,'protocol':'original synthetic train/validation, lightbox+sunlamp only test; no new random split'}
(view/'preparation_report.json').write_text(json.dumps(report,indent=2))
view.rename(fixed)
print(json.dumps(report,indent=2),flush=True)
print('FREE_BYTES',shutil.disk_usage(root).free,flush=True)
