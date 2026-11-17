#define _POSIX_C_SOURCE 200809L
/* Mide GQA (RoPE entero) forward/backward y imprime un hash de todas las salidas. Ver bench_head.c para compilar. */
#include "ternary_gqa.h"
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
    size_t T=argc>1?atoi(argv[1]):64,E=argc>2?atoi(argv[2]):128,NH=argc>3?atoi(argv[3]):4,NKV=argc>4?atoi(argv[4]):2;
    TernaryGQA*g=ternary_gqa_alloc(E,T,NH,NKV);
    if(!g){puts("alloc");return 1;}
    if(ternary_gqa_enable_rope_int(g,10.0f,1)!=0){puts("rope");return 1;}
    rnd(g->wq);rnd(g->wk);rnd(g->wv);rnd(g->wo); g->wo->scale=0.3f; g->wv->scale=0.2f;
    size_t qw=g->n_heads*g->head_dim,kvw=g->n_kv_heads*g->head_dim;
    TernaryVec*x=ternary_vec_alloc(T*E); for(size_t i=0;i<T*E;i++) ternary_vec_set(x,i,(int)(urand()*3)-1);
    float*out=malloc(T*E*4),*go=malloc(T*E*4),*gx=calloc(T*E,4); int32_t*raw=malloc(T*E*4);
    float*gw[4]={malloc(qw*E*4),malloc(kvw*E*4),malloc(kvw*E*4),malloc(E*qw*4)}; float gsc[4];
    for(size_t i=0;i<T*E;i++) go[i]=(float)(urand()-0.5);
    double tf=1e9,tb=1e9;
    for(int r=0;r<5;r++){
        double t0=now(); ternary_gqa_forward(g,x,out,raw); double t1=now();
        memset(gx,0,T*E*4); memset(gsc,0,16);
        ternary_gqa_backward(g,x,go,gw,gx,gsc,raw); double t2=now();
        if(t1-t0<tf){tf=t1-t0;} if(t2-t1<tb){tb=t2-t1;}
    }
    uint64_t hh=1469598103934665603ull;
    hh=fnv(hh,out,T*E*4); hh=fnv(hh,raw,T*E*4); hh=fnv(hh,gw[0],qw*E*4); hh=fnv(hh,gw[1],kvw*E*4); hh=fnv(hh,gw[2],kvw*E*4); hh=fnv(hh,gw[3],E*qw*4); hh=fnv(hh,gx,T*E*4); hh=fnv(hh,gsc,16);
    printf("T=%zu E=%zu heads=%zu kv=%zu fwd %.3f ms bwd %.3f ms hash=%016llx\n",T,E,NH,NKV,tf,tb,(unsigned long long)hh);
    return 0;
}
