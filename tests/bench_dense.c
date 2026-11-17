#define _POSIX_C_SOURCE 200809L
#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static uint64_t rs=88172645463325252ull;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (double)(rs>>11)/9007199254740992.0; }
#include "bench_pool.h"
int main(int argc,char**argv){
    bench_pool_init();
    size_t I=argc>1?atoi(argv[1]):2048,O=argc>2?atoi(argv[2]):2048,S=argc>3?atoi(argv[3]):32;
    TernaryDense*d=ternary_dense_alloc(I,O,0); d->scale=0.3f;
    for(size_t o=0;o<O;o++)for(size_t i=0;i<I;i++) ternary_dense_set_weight(d,o,i,(int)(urand()*3)-1);
    TernaryVec**xs=malloc(S*sizeof(*xs)); float**outs=malloc(S*sizeof(float*)),**go=malloc(S*sizeof(float*)),**gx=malloc(S*sizeof(float*)); int32_t**raws=malloc(S*sizeof(int32_t*));
    for(size_t s=0;s<S;s++){ xs[s]=ternary_vec_alloc(I); for(size_t i=0;i<I;i++) ternary_vec_set(xs[s],i,(int)(urand()*3)-1);
        outs[s]=malloc(O*4); raws[s]=malloc(O*4); go[s]=malloc(O*4); gx[s]=calloc(I,4); for(size_t o=0;o<O;o++) go[s][o]=(float)(urand()-0.5); }
    float*gw=calloc(O*I,4); float gs=0; TernaryCapabilities caps; memset(&caps,0,sizeof caps); caps.num_cores=1; caps.dot_impl_name="scalar";
    double tf=1e9,tb=1e9;
    for(int r=0;r<5;r++){ double t0=now(); ternary_dense_forward_block(d,S,(const TernaryVec*const*)xs,outs,raws,&caps); double t1=now();
        ternary_dense_backward_block(d,S,(const TernaryVec*const*)xs,(const float*const*)go,(const int32_t*const*)raws,gw,gx,&gs,NULL); double t2=now();
        if(t1-t0<tf)tf=t1-t0; if(t2-t1<tb)tb=t2-t1; }
    printf("dense %zux%zu S=%zu fwd %.3f ms bwd %.3f ms\n",I,O,S,tf,tb); return 0; }
