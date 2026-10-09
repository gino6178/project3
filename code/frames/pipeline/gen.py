# Gemini 2.5 Flash Image cut faces and exteriors for one object of objects.json: 9 lengthwise, 9 transverse,
# 4 exteriors, each a separate request -- a separate specimen, as the method assumes.
#   python gen.py <aiobj root> <object>
import os,sys,json,concurrent.futures as cf
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
import gen_ai
root,o=sys.argv[1],sys.argv[2]; R=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"objects.json")))[o]
d=f"{root}/{o}/raw"; e=f"{root}/{o}/exterior"; os.makedirs(d,exist_ok=True); os.makedirs(e,exist_ok=True)
jobs=[(f"{R['long']}. {gen_ai.STYLE}",100+k,f"{d}/long_{k}.png") for k in range(9)]+[(f"{R['trans']}. {gen_ai.STYLE}",200+k,f"{d}/trans_{k}.png") for k in range(9)]
jobs+=[(R["exterior"]+"."+gen_ai.EXT_STYLE,400+k,f"{e}/ext_{k}.png") for k in range(4)]
with cf.ThreadPoolExecutor(4) as ex: print(sum(r is not None for r in ex.map(lambda j: gen_ai.gen(*j),jobs)),"of",len(jobs),"images")
