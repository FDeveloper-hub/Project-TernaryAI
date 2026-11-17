#include "ternary_image.h"
#include "ternary_conv2d.h"
#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static void test_encode_image(void) {
    uint8_t pix[4] = {0, 128, 200, 138};
    TernaryVec *v = ternary_encode_image_u8(pix, 1, 4, 1, 128, 10);
    int esperado[4] = {-1, 0, 1, 0};
    for (int i = 0; i < 4; i++) assert(ternary_vec_get(v, i) == esperado[i]);
    printf("[encode imagen] OK\n");
    ternary_vec_free(v);
}

/* ---- Referencia plana (sin bits) de forward+backward+update, para
 * verificar Conv2D exactamente igual que se hizo con Dense. ---- */
typedef struct {
    int8_t *kernels;   /* out_channels*ksize */
    int8_t *bias;      /* out_channels */
    size_t ksize, out_channels;
    float scale;
} FlatConv;

static int32_t flat_dot(const int8_t *a, const int8_t *b, size_t n) {
    int32_t acc = 0;
    for (size_t i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

static void flat_forward(FlatConv *c, const int8_t *img, size_t H, size_t W, size_t Cin,
                          size_t KH, size_t KW, size_t stride,
                          float *out, int32_t *out_raw, size_t OH, size_t OW) {
    (void)H; /* alto total de la imagen: no hace falta, solo se usa el ancho W para el stride */
    int8_t patch[512];
    for (size_t oh = 0; oh < OH; oh++)
        for (size_t ow = 0; ow < OW; ow++) {
            size_t bh = oh*stride, bw = ow*stride, idx=0;
            for (size_t kh=0;kh<KH;kh++) for (size_t kw=0;kw<KW;kw++) for (size_t ci=0;ci<Cin;ci++)
                patch[idx++] = img[((bh+kh)*W+(bw+kw))*Cin+ci];
            for (size_t o = 0; o < c->out_channels; o++) {
                int32_t z = flat_dot(c->kernels + o*c->ksize, patch, c->ksize);
                if (c->bias) z += c->bias[o];
                size_t oidx = (oh*OW+ow)*c->out_channels+o;
                if (out_raw) out_raw[oidx] = z;
                out[oidx] = c->scale * (float)z;
            }
        }
}

static void flat_backward(FlatConv *c, const int8_t *img, size_t H, size_t W, size_t Cin,
                           size_t KH, size_t KW, size_t stride,
                           const float *grad_out, float *grad_k, float *grad_b, float *grad_scale,
                           const int32_t *out_raw, size_t OH, size_t OW) {
    (void)H; /* alto total de la imagen: no hace falta, solo se usa el ancho W para el stride */
    int8_t patch[512];
    for (size_t oh = 0; oh < OH; oh++)
        for (size_t ow = 0; ow < OW; ow++) {
            size_t bh = oh*stride, bw = ow*stride, idx=0;
            for (size_t kh=0;kh<KH;kh++) for (size_t kw=0;kw<KW;kw++) for (size_t ci=0;ci<Cin;ci++)
                patch[idx++] = img[((bh+kh)*W+(bw+kw))*Cin+ci];
            for (size_t o = 0; o < c->out_channels; o++) {
                size_t oidx = (oh*OW+ow)*c->out_channels+o;
                float g = grad_out[oidx];
                *grad_scale += g * (float)out_raw[oidx];
                if (grad_b) grad_b[o] += g;
                for (size_t k = 0; k < c->ksize; k++) grad_k[o*c->ksize+k] += g * (float)patch[k];
            }
        }
}

static void test_forward_backward_vs_referencia(void) {
    srand(7);
    size_t H=7, W=9, Cin=2, KH=3, KW=3, Cout=4, stride=1;
    int use_bias=1;

    TernaryConv2D *layer = ternary_conv2d_alloc(H,W,Cin,KH,KW,Cout,stride,use_bias);
    layer->scale = 0.41f;
    FlatConv flat; memset(&flat,0,sizeof(flat));
    flat.ksize = layer->kernel_size; flat.out_channels = Cout; flat.scale=layer->scale;
    flat.kernels = calloc(Cout*flat.ksize,1);
    flat.bias = calloc(Cout,1);

    for (size_t o=0;o<Cout;o++) for (size_t k=0;k<flat.ksize;k++) {
        int v=(rand()%3)-1;
        ternary_conv2d_set_weight(layer,o,k,v);
        flat.kernels[o*flat.ksize+k]=(int8_t)v;
    }
    for (size_t o=0;o<Cout;o++){ int v=(rand()%3)-1; ternary_vec_set(layer->bias,o,v); flat.bias[o]=(int8_t)v; }

    size_t n_img = H*W*Cin;
    int8_t *img = malloc(n_img);
    for (size_t i=0;i<n_img;i++) img[i]=(int8_t)((rand()%3)-1);
    TernaryVec *image = ternary_vec_alloc(n_img);
    for (size_t i=0;i<n_img;i++) ternary_vec_set(image,i,img[i]);

    size_t OH=layer->out_height, OW=layer->out_width, n_out=OH*OW*Cout;
    float *out_bits=malloc(n_out*sizeof(float)), *out_ref=malloc(n_out*sizeof(float));
    int32_t *raw_bits=malloc(n_out*sizeof(int32_t)), *raw_ref=malloc(n_out*sizeof(int32_t));

    ternary_conv2d_forward(layer, image, out_bits, raw_bits);
    flat_forward(&flat, img, H,W,Cin,KH,KW,stride, out_ref, raw_ref, OH,OW);

    int disc=0;
    for (size_t i=0;i<n_out;i++) if (fabsf(out_bits[i]-out_ref[i])>1e-4f) disc++;
    printf("[conv2d forward vs referencia] %zu salidas, discrepancias=%d\n", n_out, disc);
    assert(disc==0);

    float *grad_out = malloc(n_out*sizeof(float));
    for (size_t i=0;i<n_out;i++) grad_out[i] = ((float)(rand()%2000)/1000.0f - 1.0f);

    float *gk_bits=calloc(Cout*flat.ksize,sizeof(float)), *gk_ref=calloc(Cout*flat.ksize,sizeof(float));
    float *gb_bits=calloc(Cout,sizeof(float)), *gb_ref=calloc(Cout,sizeof(float));
    float gs_bits=0, gs_ref=0;

    ternary_conv2d_backward(layer, image, grad_out, gk_bits, gb_bits, &gs_bits, raw_bits);
    flat_backward(&flat, img, H,W,Cin,KH,KW,stride, grad_out, gk_ref, gb_ref, &gs_ref, raw_ref, OH,OW);

    disc=0;
    for (size_t i=0;i<Cout*flat.ksize;i++) if (fabsf(gk_bits[i]-gk_ref[i])>1e-3f) disc++;
    for (size_t i=0;i<Cout;i++) if (fabsf(gb_bits[i]-gb_ref[i])>1e-3f) disc++;
    if (fabsf(gs_bits-gs_ref)>1e-2f) disc++;
    printf("[conv2d backward vs referencia] discrepancias=%d (grad_scale bits=%.4f ref=%.4f)\n", disc, gs_bits, gs_ref);
    assert(disc==0);

    /* Update (regla R2): cada filtro es una fila; Conv2D debe dar EXACTAMENTE el mismo estado (pesos, contadores, bias)
     * que una Dense con los mismos pesos a la que se le aplican los mismos gradientes por fila. */
    TernaryDense *ref = ternary_dense_alloc(flat.ksize, Cout, 1);
    for (size_t o=0;o<Cout;o++) {
        for (size_t k=0;k<flat.ksize;k++) ternary_dense_set_weight(ref,o,k,ternary_conv2d_get_weight(layer,o,k));
        ternary_vec_set(ref->bias,o,ternary_vec_get(layer->bias,o));
    }
    memset(layer->counter_kernels,0,Cout*flat.ksize); memset(layer->counter_bias,0,Cout);
    int discrepancias_update = 0;
    for (int step = 0; step < 500; step++) {
        float *g2 = malloc(n_out*sizeof(float));
        for (size_t i=0;i<n_out;i++) g2[i] = sinf((float)step*0.05f+(float)i*0.7f);
        float *gk2 = calloc(Cout*flat.ksize,sizeof(float));
        float *gb2 = calloc(Cout,sizeof(float));
        float gs2 = 0;
        ternary_conv2d_forward(layer, image, out_bits, raw_bits);
        ternary_conv2d_backward(layer, image, g2, gk2, gb2, &gs2, raw_bits);
        for (size_t o=0;o<Cout;o++) {
            ternary_conv2d_apply_kernel_gradients(layer,o,gk2+o*flat.ksize);
            ternary_dense_apply_row_gradients(ref,o,gk2+o*flat.ksize);
        }
        ternary_conv2d_apply_bias_gradients(layer, gb2);
        ternary_dense_apply_bias_gradients(ref, gb2);
        ternary_conv2d_apply_scale_gradient(layer, gs2, 0.02f);
        for (size_t i=0;i<Cout*flat.ksize;i++) {
            if (ternary_conv2d_get_weight(layer, i/flat.ksize, i%flat.ksize) != ternary_dense_get_weight(ref, i/flat.ksize, i%flat.ksize)) discrepancias_update++;
            if (layer->counter_kernels[i] != ref->counter_W[i]) discrepancias_update++;
        }
        for (size_t o=0;o<Cout;o++) {
            if (ternary_vec_get(layer->bias,o) != ternary_vec_get(ref->bias,o)) discrepancias_update++;
            if (layer->counter_bias[o] != ref->counter_b[o]) discrepancias_update++;
        }
        free(g2); free(gk2); free(gb2);
    }
    printf("[conv2d update R2 vs Dense, 500 pasos] discrepancias=%d\n", discrepancias_update);
    assert(discrepancias_update == 0);
    ternary_dense_free(ref);

    printf("TODAS LAS PRUEBAS DE IMAGENES/CONV2D PASARON\n");

    free(img); free(out_bits); free(out_ref); free(raw_bits); free(raw_ref); free(grad_out);
    free(gk_bits); free(gk_ref); free(gb_bits); free(gb_ref);
    free(flat.kernels); free(flat.bias);
    ternary_vec_free(image);
    ternary_conv2d_free(layer);
}

int main(void) {
    test_encode_image();
    test_forward_backward_vs_referencia();
    return 0;
}
