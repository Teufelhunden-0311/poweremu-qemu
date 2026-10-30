#!/usr/bin/env python3
"""Compile real AGP functions with a deterministic address-space test double.
No production refactor. Fails closed if a named function cannot be extracted.
--mutants runs targeted mutations in temporary files, never the repository.
"""
import argparse, os, pathlib, re, shlex, subprocess, tempfile
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path);p.add_argument('--mutants',action='store_true');a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
source=a.source or here.parents[1]/'hw/display/ppc_mac_gpu.c'
s=source.read_text()
def function(name, text=s):
 m=re.search(r'^(?:static )?(?:bool|void|int|uint8_t)[ \t*]+\b'+name+r'\([^;{}]*\)\s*\{',text,re.M)
 if not m:raise RuntimeError('missing function '+name)
 # At this pin every function's closing brace is at column zero.
 end=text.index('\n}',m.end())+2
 return text[m.start():end]+'\n'
names=['r300_span','r300_fb_range','r300_in_fb','r300_in_agp','r300_in_aic','r300_sys_span',
       'r200_agp_tc_flush','r200_agp_cached','r200_agp_page','r300_build_vram_range','r300_build_agp_range','r300_build_parked']
drawpath = source.parent/'r300/r300_draw.c'
if not drawpath.exists(): drawpath=source.parent/'hw/display/r300/r300_draw.c'
draw=drawpath.read_text()
code='\n'.join(function(n,draw) for n in ['r300_capture_read','r300_draw_parked_begin','r300_draw_parked_end'])+'\n'+'\n'.join(function(n) for n in names)
mutations={
 'ignore_config':('memcmp(&cfg, &r200_agp_tc_cfg, sizeof(cfg)) ||','false ||'),
 'ignore_tag':('r200_agp_tc[i].tag != page + 1','false'),
 'fresh_walk':('page = r200_agp_cached(s, a >> 12);','page = r200_agp_page(s, a);'),
 'wrong_offset':('memcpy(dst, page + (a & 0xFFF), n);','memcpy(dst, page, n);'),
 'allow_aic_overlap':('((s->regs.aic_ctrl & 1) && a <= s->regs.aic_hi_addr &&','(false && a <= s->regs.aic_hi_addr &&'),
 'allow_pci_only':('!r300_sys_span(s, a, n) || !r300_in_agp(s, a, n) ||\n            ((s->regs.aic_ctrl & 1)', '!r300_sys_span(s, a, n) ||\n            ((false)'),
 'capture_wrap':('(uint64_t)a + len > 0x100000000ull','false'),
 'live_P2_read':('memcpy(dst, c->buf + at + (gpu_addr - r->gpu), len);','memcpy(dst, A + (gpu_addr - r->gpu), len);'),
 'ignore_cancel':('if (!cp_wait_relock(w)) {','if (cp_wait_relock(w) && false) {'),
 'ignore_redo':('if (redo) {','if (redo && false) {'),
 'accept_alloc_failure':('if (cap.plan.total && !cap.buf) {','if (cap.plan.total && !cap.buf) { *ok = true; return 1; } if (false) {'),
}
with tempfile.TemporaryDirectory(prefix='agpcap-') as tmp:
 d=pathlib.Path(tmp);header=d/'agpcap_under_test.h';exe=d/'test'
 cc=shlex.split(os.environ.get('CC','cc'))
 def run(text):
  header.write_text(text)
  subprocess.run(cc+['-std=gnu11','-O1','-g','-Wall','-Wextra','-Werror','-Wno-unused-function','-Wno-unused-variable','-Wno-unused-but-set-variable','-fsanitize=undefined','-fno-sanitize-recover=all','-I',str(d),str(here/'test_agpcap.c'),'-o',str(exe)],check=True)
  return subprocess.run([str(exe)],capture_output=True,text=True)
 r=run(code)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 print(r.stdout,end='')
 if a.mutants:
  for name,(old,new) in mutations.items():
   # Some source strings occur in both peek and ordinary path: mutate peek only.
   if name in ('ignore_config','ignore_tag'):
    prefix=code.index('static uint8_t *r200_agp_cached');end=code.index('static uint8_t *r200_agp_page',prefix)
    body=code[prefix:end];assert body.count(old)==1,(name,body.count(old))
    mutant=code[:prefix]+body.replace(old,new)+code[end:]
   else:
    assert code.count(old)==1,(name,code.count(old));mutant=code.replace(old,new)
   r=run(mutant)
   if r.returncode==0:raise RuntimeError('MISSED '+name)
   print('caught',name)
