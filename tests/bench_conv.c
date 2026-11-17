#define _POSIX_C_SOURCE 200809L
/* Mide Conv2D forward/backward y imprime un hash de todas las salidas (bits identicos). Ver bench_head.c para compilar. */
#include "ternary_conv2d.h"
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
    size_t H=argc>1?atoi(argv[1]):28,W=argc>2?atoi(argv[2]):28,C=argc>3?atoi(argv[3]):1,KH=argc>4?atoi(argv[4]):3,KW=argc>5?atoi(argv[5]):3,O=argc>6?atoi(argv[6]):32,ST=argc>7?atoi(argv[7]):1;
    TernaryConv2D*l=ternary_conv2d_alloc(H,W,C,KH,KW,O,ST,1);
    for(size_t o=0;o<O;o++)for(size_t k=0;k<l->kernel_size;k++) ternary_conv2d_set_weight(l,o,k,(int)(urand()*3)-1);
    for(size_t o=0;o<O;o++) ternary_vec_set(l->bias,o,(int)(urand()*3)-1);
    l->scale=0.37f;
    TernaryVec*img=ternary_vec_alloc(H*W*C);
    for(size_t i=0;i<H*W*C;i++) ternary_vec_set(img,i,(int)(urand()*3)-1);
    size_t no=l->out_height*l->out_width*O;
    float*out=malloc(no*4),*g=malloc(no*4),*gk=malloc(O*l->kernel_size*4),*gb=calloc(O,4); int32_t*raw=malloc(no*4);
    for(size_t i=0;i<no;i++) g[i]=(float)(urand()-0.5)*(urand()<0.3?0.f:1.f);
    double tf=1e9,tb=1e9; float gs=0;
    for(int r=0;r<5;r++){
        double t0=now(); ternary_conv2d_forward(l,img,out,raw); double t1=now();
        memset(gk,0,O*l->kernel_size*4); memset(gb,0,O*4); gs=0;
        ternary_conv2d_backward(l,img,g,gk,gb,&gs,raw); double t2=now();
        if(t1-t0<tf){tf=t1-t0;} if(t2-t1<tb){tb=t2-t1;}
    }
    uint64_t hh=1469598103934665603ull;
    hh=fnv(hh,out,no*4); hh=fnv(hh,raw,no*4); hh=fnv(hh,gk,O*l->kernel_size*4); hh=fnv(hh,gb,O*4); hh=fnv(hh,&gs,4);
    /* backward sin z_raw */
    memset(gk,0,O*l->kernel_size*4); memset(gb,0,O*4); gs=0;
    ternary_conv2d_backward(l,img,g,gk,gb,&gs,NULL);
    hh=fnv(hh,gk,O*l->kernel_size*4); hh=fnv(hh,gb,O*4); hh=fnv(hh,&gs,4);
    printf("img %zux%zux%zu k %zux%zu o %zu s %zu  fwd %.3f ms  bwd %.3f ms  hash=%016llx\n",H,W,C,KH,KW,O,ST,tf,tb,(unsigned long long)hh);
    return 0;
}
