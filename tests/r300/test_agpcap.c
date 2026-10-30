/* AI-generated host contract witnesses. Actual production functions extracted
 * by test_agpcap.py. Device/RCU/BQL scheduling are doubles, NOT VM evidence. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define MiB (1024*1024)
#define R300_BUILD_CAPTURE_MAX (4*MiB)
#define RCU_READ_LOCK_GUARD() ((void)0)
#define MEMTXATTRS_UNSPECIFIED 0
#define P 0x10000u
#define Q (P+1024u)
typedef uint64_t hwaddr;
typedef struct { bool ram; uint8_t *ptr; } MemoryRegion;
typedef struct { struct { uint32_t mc_fb_location,mc_agp_location,aic_ctrl,aic_lo_addr,aic_hi_addr,aic_pt_base; } regs;
 uint64_t vram_size; uint32_t r300_aic_pt_base; void *r3_dump,*r3; int r3_arrays; uint8_t *vram_host; MemoryRegion vram;
} PPCMacGPUState;
typedef struct { uint32_t agp_loc,aic_ctrl,aic_lo,aic_hi,aic_pt,r300_pt,bridge_gen; } R200AgpCfg;
static struct { uint32_t tag; uint8_t *host; } r200_agp_tc[1024];
static R200AgpCfg r200_agp_tc_cfg;
typedef struct { uint32_t gpu,len; } R300FetchRange;
typedef struct { unsigned n; uint64_t total; R300FetchRange r[4]; } R300FetchPlan;
typedef struct { R300FetchPlan plan; uint8_t *buf; } R300Capture;
typedef struct { int unused; } R300Indices;
typedef struct { uint8_t bytes[8192]; unsigned n; } R300DrawPacket;
static bool r300_build_unlock=true,r300_build_unlock_agp=true,r300_build_debug,r300_vfetch=true;
static uint32_t generation;static unsigned walks,mmio_reads,data_translates;
static uint8_t A[4096],B[4096],Cpage[4096],VRAM[65536];
static uint8_t *pte_p;static bool pte_mmio,data_ram=true;
static int address_space_memory;
static uint32_t uninorth_get_agp_gart_gen(void){return generation;}
static bool ppc_mac_gpu_agp_translate(PPCMacGPUState *s,uint32_t a,hwaddr *phys)
{
 (void)s;walks++;if(pte_mmio){mmio_reads++;return false;}
 *phys=(a>>12)==P?0:4096;return true;
}
static bool ppc_mac_gpu_gart_translate(PPCMacGPUState *s,uint32_t a,hwaddr *phys)
{(void)s;(void)a;(void)phys;return false;}
static MemoryRegion *address_space_translate(void *as,hwaddr phys,hwaddr *xlat,hwaddr *len,bool wr,int attrs)
{
 static MemoryRegion mr;(void)as;(void)wr;(void)attrs;(void)len;
 data_translates++;*xlat=0;mr.ram=data_ram;mr.ptr=phys==0?pte_p:Cpage;return &mr;
}
static bool memory_region_is_ram(MemoryRegion *m){return m->ram;}
static void *memory_region_get_ram_ptr(MemoryRegion *m){return m->ptr;}
static R300FetchPlan planned;static bool plan_ok=true,alloc_fail,parked,cancel;
static bool build_parked,build_needs_protected;
#define redo build_needs_protected
static unsigned allocations,frees,builds,unlock_calls;static int action;
static bool cp_can_unlock(void){return true;}
static bool r300_endian_model(void){return true;}
static bool r300_draw_fetch_plan(void *st,int *arr,uint32_t op,const uint32_t *d,uint32_t ndw,const R300Indices *idx,uint64_t cap,R300FetchPlan *out)
{(void)st;(void)arr;(void)op;(void)d;(void)ndw;(void)idx;*out=planned;return plan_ok && out->total<=cap;}
static void *g_try_malloc(size_t n){allocations++;return alloc_fail?NULL:malloc(n);}
static void g_free(void *p){if(p)frees++;free(p);}
static int cp_wait_unlock(void){assert(!parked);parked=true;unlock_calls++;return 1;}
static bool cp_wait_relock(int w){assert(w==1 && parked);parked=false;return !cancel;}
void r300_draw_parked_begin(void);
bool r300_draw_parked_end(void);
static void r300_draw_free(R300DrawPacket *p){memset(p,0,sizeof(*p));}
bool r300_capture_read(void *,uint32_t,void *,uint32_t);
static bool r300_draw_build(void *st,int *arr,uint32_t op,const uint32_t *d,uint32_t ndw,
 bool (*readfn)(void *,uint32_t,void *,uint32_t),void *opaque,R300DrawPacket *pkt,const char **err)
{
 (void)st;(void)arr;(void)op;(void)d;(void)ndw;(void)err;assert(parked);builds++;
 if(action==1){memset(A,0xee,sizeof(A));pte_p=B;generation++;memset(r200_agp_tc,0,sizeof(r200_agp_tc));}
 if(action==4){pte_p=B;generation++;}
 if(action==2)cancel=true;
 if(action==3){redo=true;return false;}
 pkt->n=0;
 for(unsigned i=0;i<planned.n;i++){
  assert(pkt->n+planned.r[i].len<=sizeof(pkt->bytes));
  if(!readfn(opaque,planned.r[i].gpu,pkt->bytes+pkt->n,planned.r[i].len))return false;
  pkt->n+=planned.r[i].len;
 }
 return true;
}
static bool r300_draw_build_indexed(void *st,int *arr,uint32_t d,const R300Indices *idx,
 bool (*readfn)(void *,uint32_t,void *,uint32_t),void *opaque,R300DrawPacket *pkt,const char **err)
{(void)idx;return r300_draw_build(st,arr,0,&d,1,readfn,opaque,pkt,err);}
#include "agpcap_under_test.h"
static PPCMacGPUState s;
static void cfg(void){r200_agp_tc_cfg=(R200AgpCfg){s.regs.mc_agp_location,s.regs.aic_ctrl,s.regs.aic_lo_addr,s.regs.aic_hi_addr,s.regs.aic_pt_base,s.r300_aic_pt_base,generation};}
static void hit(uint32_t page,uint8_t *ptr){r200_agp_tc[page&1023].tag=page+1;r200_agp_tc[page&1023].host=ptr;}
static void setup(void)
{
 memset(&s,0,sizeof(s));s.regs.mc_fb_location=0;s.vram_size=sizeof(VRAM);s.vram_host=VRAM;s.vram.ptr=VRAM;
 s.regs.mc_agp_location=0x1fff1000;generation=7;cfg();r200_agp_tc_flush();
 for(unsigned i=0;i<4096;i++){A[i]=(i*7+3)%251;B[i]=0xbb;Cpage[i]=0xcc;}
 pte_p=A;pte_mmio=false;data_ram=true;walks=mmio_reads=data_translates=0;
 r300_build_unlock_agp=true;alloc_fail=false;plan_ok=true;redo=false;cancel=false;action=0;
 hit(P,A);hit(P+1,Cpage);planned=(R300FetchPlan){.n=1,.total=64,.r={{P<<12,64}}};
}
static bool range(uint32_t a,uint32_t n,uint8_t *out){R300FetchRange r={a,n};return r300_build_agp_range(&s,&r,out);}
static void unchanged_reject(uint32_t a,uint32_t n)
{
 unsigned char old[sizeof(r200_agp_tc)];R200AgpCfg oldcfg=r200_agp_tc_cfg;
 memcpy(old,r200_agp_tc,sizeof(old));unsigned w=walks,m=mmio_reads,t=data_translates;
 assert(!range(a,n,NULL));assert(!memcmp(old,r200_agp_tc,sizeof(old)));
 assert(!memcmp(&oldcfg,&r200_agp_tc_cfg,sizeof(oldcfg)));assert(w==walks && m==mmio_reads && t==data_translates);
}
static int build(R300DrawPacket *pkt){const char *err=NULL;bool ok=false;uint32_t d=0;return r300_build_parked(&s,0,&d,1,NULL,pkt,&err,&ok);}
int main(void)
{
 uint8_t out[8192],expect[64];R300DrawPacket pkt;setup();
 /* Cached A wins over changed PTE B; ordinary hit agrees without walking. */
 pte_p=B;assert(r200_agp_page(&s,P<<12)==A);assert(range((P<<12)+37,64,out));assert(!memcmp(out,A+37,64));assert(walks==0);
 assert(range((P<<12)+4090,20,out));assert(!memcmp(out,A+4090,6));assert(!memcmp(out+6,Cpage,14));
 uint32_t *fields[]={&s.regs.mc_agp_location,&s.regs.aic_ctrl,&s.regs.aic_lo_addr,&s.regs.aic_hi_addr,&s.regs.aic_pt_base,&s.r300_aic_pt_base,&generation};
 for(unsigned i=0;i<ARRAY_SIZE(fields);i++){setup();(*fields[i])++;unchanged_reject(P<<12,64);}
 setup();r200_agp_tc_flush();unchanged_reject(P<<12,64);
 setup();generation++;pte_p=B;unchanged_reject(P<<12,64);assert(r200_agp_page(&s,P<<12)==B);assert(walks==1);assert(range(P<<12,64,out));assert(out[0]==0xbb);
 /* Collision P,Q,P: the ordinary cache is order dependent after a stale PTE. */
 setup();pte_p=B;assert(r200_agp_page(&s,P<<12)==A);unchanged_reject(Q<<12,4);
 assert(r200_agp_page(&s,Q<<12)==Cpage);unchanged_reject(P<<12,4);assert(r200_agp_page(&s,P<<12)==B);
 setup();r200_agp_tc_flush();pte_mmio=true;unchanged_reject(P<<12,64);assert(!r200_agp_page(&s,P<<12));assert(mmio_reads==1);
 setup();r200_agp_tc_flush();data_ram=false;unchanged_reject(P<<12,64);assert(!r200_agp_page(&s,P<<12));assert(data_translates==1);
 setup();r200_agp_tc[P&1023].host=NULL;unchanged_reject(P<<12,64);
 setup();s.regs.mc_agp_location=0;s.regs.aic_ctrl=1;s.regs.aic_lo_addr=P<<12;s.regs.aic_hi_addr=(P<<12)+4095;cfg();unchanged_reject(P<<12,64);
 setup();s.regs.aic_ctrl=1;s.regs.aic_lo_addr=(P<<12)+32;s.regs.aic_hi_addr=(P<<12)+127;cfg();unchanged_reject(P<<12,64);
 setup();r200_agp_tc[(P+1)&1023].tag=0;unchanged_reject((P<<12)+4090,20);
 setup();unchanged_reject(P<<12,0);r300_build_unlock_agp=false;unchanged_reject(P<<12,4);
 setup();s.regs.mc_agp_location=0xffff0000;s.vram_size=0;cfg();hit(0xfffff,A);hit(0,Cpage);assert(range(0xfffffffc,4,out));unchanged_reject(0xfffffffc,8);
 /* Actual protected/parked/relock wrapper, deterministic scheduling doubles. */
 setup();memcpy(expect,A,64);action=1;unsigned w=walks;assert(build(&pkt)==1);assert(!memcmp(pkt.bytes,expect,64));assert(walks==w);assert(generation==8);
 setup();memcpy(expect,A,64);action=4;assert(build(&pkt)==1);assert(!memcmp(pkt.bytes,expect,64));assert(generation==8);
 setup();memset(VRAM,0x91,sizeof(VRAM));planned.n=2;planned.total=80;planned.r[1]=(R300FetchRange){32,16};
 assert(build(&pkt)==1 && pkt.n==80);assert(!memcmp(pkt.bytes,A,64));assert(!memcmp(pkt.bytes+64,VRAM+32,16));
 setup();action=2;assert(build(&pkt)==-1);assert(pkt.n==0);
 setup();action=3;assert(build(&pkt)==0);assert(pkt.n==0);
 setup();alloc_fail=true;unsigned u=unlock_calls;assert(build(&pkt)==0);assert(unlock_calls==u);
 setup();planned.total=R300_BUILD_CAPTURE_MAX+1;u=unlock_calls;assert(build(&pkt)==0);assert(unlock_calls==u);
 setup();plan_ok=false;assert(build(&pkt)==0);
 assert(!parked);puts("AGP actual-function contract witnesses: PASS (host doubles; not VM scheduling)");return 0;
}
