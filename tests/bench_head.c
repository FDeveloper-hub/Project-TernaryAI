/* Compilar: cc -std=c11 -O3 -Iinclude src/ternary_{core,dense,conv2d,image,dispatch,threadpool,tensor,embedding,layer,activation,model_io,stack,rng,softmax,gqa,rmsnorm,swiglu,head,loss,residual,tokenizer,json,dataset,opencl,vulkan}.c tests/bench_head.c -lm -ldl -lpthread */
#define _POSIX_C_SOURCE 200809L
/* Mide LMHead forward/backward y imprime un hash de TODAS las salidas (para comprobar bits identicos). */
#include "ternary_head.h"
#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static uint64_t rs=88172645463325252ull;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (double)(rs>>11)/9007199254740992.0; }
static uint64_t fnv(uint64_t h,const void*p,size_t n){ const unsigned char*c=p; for(size_t i=0;i<n;i++){h^=c[i];h*=1099511628211ull;} return h; }
#include "bench_pool.h"
int main(int argc,char**argv){
    bench_pool_init();
    size_t rows=argc>1?atoi(argv[1]):64, dim=argc>2?atoi(argv[2]):128, voc=argc>3?atoi(argv[3]):4096;
    TernaryHead*h=ternary_head_alloc(rows,dim,voc);
    for(size_t v=0;v<voc;v++)for(size_t i=0;i<dim;i++) ternary_dense_set_weight(h->w,v,i,(int)(urand()*3)-1);
    h->w->scale=0.37f;
    TernaryVec*x=ternary_vec_alloc(rows*dim);
    for(size_t i=0;i<rows*dim;i++) ternary_vec_set(x,i,(int)(urand()*3)-1);
    float*out=malloc(rows*voc*4),*g=malloc(rows*voc*4),*gw=malloc(voc*dim*4),*gx=calloc(rows*dim,4);
    for(size_t i=0;i<rows*voc;i++) g[i]=(float)(urand()-0.5)*(urand()<0.3?0.f:1.f);
    int reps=5; double tf=1e9,tb=1e9; float gs=0;
    for(int r=0;r<reps;r++){
        double t0=now(); ternary_head_forward(h,x,out,NULL); double t1=now();
        memset(gx,0,rows*dim*4); gs=0;
        ternary_head_backward(h,x,g,gw,gx,&gs); double t2=now();
        if(t1-t0<tf){tf=t1-t0;} if(t2-t1<tb){tb=t2-t1;}
    }
    uint64_t hh=1469598103934665603ull;
    hh=fnv(hh,out,rows*voc*4); hh=fnv(hh,h->z,rows*voc*4); hh=fnv(hh,gw,voc*dim*4); hh=fnv(hh,gx,rows*dim*4); hh=fnv(hh,&gs,4);
    printf("rows=%zu dim=%zu vocab=%zu  fwd %.3f ms  bwd %.3f ms  hash=%016llx\n",rows,dim,voc,tf,tb,(unsigned long long)hh);
    return 0;
}
