#define _POSIX_C_SOURCE 200809L
/* Mide SwiGLU forward/backward y imprime un hash de todas las salidas (bits identicos). Ver bench_head.c para compilar. */
#include "ternary_swiglu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static uint64_t rs=88172645463325252ull;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (double)(rs>>11)/9007199254740992.0; }
static uint64_t fnv(uint64_t h,const void*p,size_t n){ const unsigned char*c=p; for(size_t i=0;i<n;i++){h^=c[i];h*=1099511628211ull;} return h; }
static void rnd(TernaryDense*d){ for(size_t o=0;o<d->output_dim;o++)for(size_t i=0;i<d->input_dim;i++) ternary_dense_set_weight(d,o,i,(int)(urand()*3)-1); }
#include "bench_pool.h"
int main(int argc,char**argv){
    bench_pool_init();
    size_t R=argc>1?atoi(argv[1]):64,E=argc>2?atoi(argv[2]):128,H=argc>3?atoi(argv[3]):512; int mode=argc>4?atoi(argv[4]):0;
    TernarySwiGLU*s=ternary_swiglu_alloc(R,E,H,mode,0.5f);
    rnd(s->wg);rnd(s->wu);rnd(s->wd); s->wg->scale=0.05f; s->wu->scale=0.07f; s->wd->scale=0.3f; ternary_swiglu_refresh_tables(s);
    TernaryVec*x=ternary_vec_alloc(R*E); for(size_t i=0;i<R*E;i++) ternary_vec_set(x,i,(int)(urand()*3)-1);
    float*out=malloc(R*E*4),*g=malloc(R*E*4),*gx=calloc(R*E,4); int32_t*raw=malloc(R*E*4);
    float*gw[3]={malloc(H*E*4),malloc(H*E*4),malloc(E*H*4)}; float gsc[3];
    for(size_t i=0;i<R*E;i++) g[i]=(float)(urand()-0.5);
    double tf=1e9,tb=1e9;
    for(int r=0;r<5;r++){
        double t0=now(); ternary_swiglu_forward(s,x,out,raw); double t1=now();
        memset(gx,0,R*E*4); gsc[0]=gsc[1]=gsc[2]=0;
        ternary_swiglu_backward(s,x,g,gw,gx,gsc); double t2=now();
        if(t1-t0<tf){tf=t1-t0;} if(t2-t1<tb){tb=t2-t1;}
    }
    uint64_t hh=1469598103934665603ull;
    hh=fnv(hh,out,R*E*4); hh=fnv(hh,raw,R*E*4); for(int m=0;m<3;m++) hh=fnv(hh,gw[m],H*E*4); hh=fnv(hh,gx,R*E*4); hh=fnv(hh,gsc,12);
    printf("rows=%zu dim=%zu hidden=%zu mode=%d fwd %.3f ms bwd %.3f ms hash=%016llx\n",R,E,H,mode,tf,tb,(unsigned long long)hh);
    return 0;
}
