#ifndef TERNARY_AVX512_H
#define TERNARY_AVX512_H
/*
 * Nucleos AVX-512 (x86-64). Se eligen EN TIEMPO DE EJECUCION: ternary_avx512_enabled() es 1 solo si la CPU tiene
 * AVX512F/BW/DQ/VL (y el sistema operativo guarda el estado de los registros zmm) y no se pidio apagarlo con
 * TERNARYAI_NO_AVX512=1. No hace falta ningun flag de compilacion (-mavx512*): las funciones llevan su propio
 * atributo de arquitectura, asi que el mismo binario corre en cualquier CPU. En otras arquitecturas (ARM) las
 * funciones existen solo como esqueletos que nunca se llaman.
 *
 * Regla de todos los nucleos: el resultado es IDENTICO BIT A BIT al del camino escalar de siempre (cada elemento
 * conserva su orden de suma; nunca se usa FMA, que cambiaria un redondeo), salvo donde el comentario de la funcion
 * diga lo contrario.
 */
#include <stddef.h>
#include <stdint.h>

int ternary_avx512_enabled(void);

/* Separa los planos mask/sign de n elementos en dos arreglos de ceil(n/16) mascaras de 16 bits: plus = (mask & ~sign)
 * (el elemento vale +1) y minus = (mask & sign) (vale -1); los bits >= n quedan en 0. Los planos son little-endian y
 * deben tener ceil(n/16) medias palabras legibles. Es C portable (no necesita AVX-512). */
void ternary_avx512_split_planes(uint16_t *plus, uint16_t *minus, size_t n, const uint64_t *mask, const uint64_t *sign);

/* grad_w: base[o*ld + i] += sum_s gT[o*S + s] * x[s][i] para o < outd, i < n, con x[s][i] en {-1,0,+1} dado por las
 * mascaras xp/xn (S arreglos de nt16 = ceil(n/16) mascaras, ver split_planes). Suma en orden de s ascendente por
 * elemento, solo mascara-suma / mascara-resta de g (sin multiplicar ni FMA): mismo resultado que el bucle
 * a += g * x salvo el signo de un cero. */
void ternary_avx512_gradw(float *base, size_t ld, size_t outd, size_t n, size_t S, const float *gT,
                          const uint16_t *xp, const uint16_t *xn);

/* grad_x: gx[s][i] += sum_o gs[s*outd + o] * w[o][i], con w[o][i] en {-1,0,+1} dado por wp/wn (outd arreglos de
 * nt16 mascaras). Orden de o ascendente por elemento; mismo resultado que el bucle escalar salvo el signo de un
 * cero. */
void ternary_avx512_gradx(float *const *gx, size_t S, size_t outd, size_t n, const float *gs,
                          const uint16_t *wp, const uint16_t *wn);

/* Forward entero con VNNI (vpdpbusd). ternary_avx512_vnni_enabled(): AVX-512 mas VNNI (TERNARYAI_NO_VNNI=1 lo apaga).
 * expand_i8: n pesos (mascara/signo) -> n int8 en {-1,0,1}. ternary_vnni_forward: P posiciones q[p*qs ..] (u8, in entradas) contra
 * pesos wt[(k*Opad + o)*4 + j] = w[o][4k+j] (Opad = O redondeado a 16, ceros de relleno; wsum[o] = sum_i w[o][i] y bias[o] o NULL,
 * ambos de Opad enteros): z = sum(q*w) - zp[p]*wsum[o] + bias[o]; out[p*os + o] = (scale*sx[p]) * (float)z; raw[p*rs + o] = z (raw
 * puede ser NULL). Mismo resultado exacto que ternary_avx512_dot_u8 fila por fila. */
int ternary_avx512_vnni_enabled(void);
void ternary_avx512_expand_i8(const uint64_t *mask, const uint64_t *sign, size_t n, int8_t *out);
void ternary_vnni_forward(const int8_t *wt, const int32_t *wsum, const int32_t *bias, size_t Opad, size_t O, size_t in, size_t P,
                          const uint8_t *q, size_t qs, const float *sx, const int32_t *zp, float scale,
                          float *out, size_t os, int32_t *raw, size_t rs);

/* ternary_avx512_gradx con g = gs[s*outd + o] * scale calculado dentro (mismo producto float que el camino por fila). */
void ternary_avx512_gradx_scaled(float *const *gx, size_t S, size_t outd, size_t n, const float *g, float scale,
                                 const uint16_t *wp, const uint16_t *wn);

/* terms[p*ts + o] = (g == 0 ? 0 : g * (sx[p] * (float)z[p*zs + o])) para p < P, o en [o0, o1): igual bit a bit que el bucle escalar. */
void ternary_avx512_scale_terms(float *terms, size_t ts, const float *g, size_t gs, const float *sx, const int32_t *z,
                                size_t zs, size_t P, size_t o0, size_t o1);

/* gw[o*n + i] = fma(g[p*gs + o], x[p*xs + i], gw[o*n + i]) para o en [o0, o1), i < n, p ascendente (g == 0 se salta);
 * multiplicacion y suma FUSIONADAS (un solo redondeo): igual bit a bit que el bucle escalar con fmaf, que es lo que corre
 * en NEON (en ARMv7 compile con -mfpu=neon-vfpv4 para que fmaf sea una instruccion y no una llamada de software). */
void ternary_avx512_outer_acc_rows(float *gw, size_t n, size_t o0, size_t o1, size_t P, const float *g, size_t gs,
                                   const float *x, size_t xs);

/* Empaqueta n floats en planos ternarios: +1 si x > thr, -1 si x < -thr, 0 en otro caso (NaN da 0). Con thr = 0 es la
 * activacion "sign"; con thr = factor*media|x| es "band". Escribe solo los bits < n (los demas no se tocan). Identico
 * bit a bit al bucle ternary_vec_set de siempre. */
void ternary_avx512_pack_thr(const float *x, size_t n, float thr, uint64_t *mask, uint64_t *sign);

/* sum |x[i]| en float. El orden de suma es por carriles (distinto al secuencial): difiere del escalar en el ultimo bit. */
float ternary_avx512_sum_abs(const float *x, size_t n);

/* RMSNorm con ganancia ternaria (gp/gm = planos separados con split_planes sobre D). La suma de cuadrados y el
 * producto punto van en double con orden por carriles (puede diferir del escalar en el ultimo bit del double; el
 * resultado final se redondea a float). La salida elemento a elemento es identica a la escalar. */
void ternary_avx512_rmsnorm_forward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                    const uint16_t *gm, const float *x, float *out);
/* grad_gain_out debe venir en cero; grad_x_out puede ser NULL; *gscale += sum dL/ds. */
void ternary_avx512_rmsnorm_backward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                     const uint16_t *gm, const float *x, const float *go, float *grad_gain_out,
                                     float *grad_x_out, double *gscale);

/* Backward de UNA muestra para una fila de salida o: grad_row[i] (+)= g * x[i]  y  gx[i] += gs * w[i], i < in, con x y
 * w en {-1,0,+1} dados por las mascaras xp/xn y wp/wn (ceil(in/16) cada una). Multiplica de verdad (sin FMA): identico
 * bit a bit al bucle escalar. accumulate=0 sobrescribe grad_row. */
void ternary_avx512_bwd1_row(size_t in, const uint16_t *xp, const uint16_t *xn, const uint16_t *wp, const uint16_t *wn,
                             float g, float gs, float *grad_row, float *gx, int accumulate);

/* Desempaqueta n trits que empiezan en el bit bit0 de los planos (mask/sign, nbits_total bits validos en total) a
 * out[i] = scale * (float)w y, si raw != NULL, raw[i] = w. Identico bit a bit a ternary_vec_get + multiplicar. */
void ternary_avx512_unpack_row(const uint64_t *mask, const uint64_t *sign, size_t nbits_total, size_t bit0, size_t n,
                               float scale, float *out, int32_t *raw);

/* Puntajes enteros de atencion: out[j] = clamp((sum_d a[d]*kb[j*kstride + d]) >> shift, -lim, lim) para j < count,
 * con suma en int64 (exacta; mismo resultado que el bucle escalar). */
void ternary_avx512_scores_i32(const int32_t *a, const int32_t *kb, size_t kstride, size_t H, size_t count, int shift,
                               int32_t lim, int32_t *out);

/* gq[d] += ds[j]*(float)kb[j*kstride+d] y gk[j*kstride+d] += ds[j]*(float)qv[d], j ascendente, d < H. Identico bit a bit. */
void ternary_avx512_qk_grad(float *gq, const int32_t *qv, const int32_t *kb, float *gk, size_t kstride, size_t H,
                            size_t count, const float *ds);

/* dst[j*stride + d] += p[j] * ga[d] para j < count, d < H. Identico bit a bit. */
void ternary_avx512_outer_add(float *dst, size_t stride, size_t H, size_t count, const float *p, const float *ga);

/* ---- Atencion "flash" ternaria (ver ternary_gqa.c) ----------------------------------------------------------
 * Todos operan sobre las claves en bloques de 16 (K y V traspuestas: kT[d*ld + j], vT[d*ld + j]; ld multiplo de 16
 * y las columnas >= n legibles) y sobre un GRUPO de G cabezas de consulta que comparten K/V (GQA). Identicos bit a
 * bit a las funciones escalares attn_forward_flash / attn_backward_flash. */

/* sc[b*scs + j] = clamp((sum_d q[b][d]*kT[d*ld + j]) >> shift, -lim, lim) para j < n (int64 exacto) y m_out[b] = max_j. */
void ternary_avx512_attn_scores(const int32_t *const *q, size_t G, const int32_t *kT, size_t ld, size_t H, size_t n,
                                int shift, int32_t lim, int32_t *sc, size_t scs, int32_t *m_out);

/* Forward de la fila: e_j = tab[min(m - s_j, K0)] (se guarda en e[b*es + j]), suma por 16 carriles en arbol fijo,
 * out[b][d] = (sum_j e_j * v[j*vs + d]) * (1/suma) con j ascendente, inv_out[b] = 1/suma. out = G filas de H, paso H. */
void ternary_avx512_attn_pv(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                            size_t n, const float *v, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out);

/* Backward de la fila, G cabezas: acumula en gq (G filas de H, paso H), gk y gv (fila j en gk[j*gs], gv[j*gs]).
 * ga/o: G filas de H (gradiente de la salida y salida de la atencion); qr: G filas de H (q_rot); kr[j*ks + d] = k_rot.
 * Trabajo: p, ds (G x es), qf (G x H floats). */
void ternary_avx512_attn_bwd(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                             const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                             const float *vT, size_t ld, float it, float gam, const int32_t *qr, const int32_t *kr,
                             size_t ks, float *gq, float *gk, float *gv, size_t gs, float *p, float *ds, size_t es,
                             float *qf);

/* Backward por bloques de R filas de consulta consecutivas (misma cuenta bit a bit que R llamadas a attn_bwd, ver ternary_avx512.c).
 * attn_bwd_pds = fase 1 de una fila (p, ds, qf). attn_bwd_rows_cap(G, H) = filas por bloque que admite la configuracion (0: no hay).
 * attn_bwd_rows: fila r usa nr[r] claves (nr no decreciente); ga: fila r en ga + r*gas; qf: (r*G + b)*H; p/ds: r*rs + b*es;
 * gq: fila r en gq + r*gqs; gk/gv: fila j en gk[j*gs]; doA = 0 omite gk y gv. */
void ternary_avx512_attn_bwd_pds(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                                 const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                                 const float *vT, size_t ld, float it, float gam, const int32_t *qr,
                                 float *p, float *ds, size_t es, float *qf);
size_t ternary_avx512_attn_bwd_rows_cap(size_t G, size_t H);
void ternary_avx512_attn_bwd_rows(size_t R, size_t G, size_t H, const size_t *nr, const float *ga, size_t gas,
                                  const float *qf, const float *p, const float *ds, size_t es, size_t rs,
                                  const int32_t *kr, size_t ks, float *gq, size_t gqs, float *gk, float *gv,
                                  size_t gs, int doA);

/* Variantes con DOS tramos de claves (ventana + sinks), ver el comentario en ternary_avx512.c: tramo A = nA claves, tramo B
 * = nB claves; indices de trabajo A en [0, spA) (spA = nA redondeado a 16) y B desde spA. vA/vB: V por filas de cada tramo;
 * vTA/vTB: V traspuesta (columna de inicio de cada tramo ya sumada); krA/krB, gkA/gkB, gvA/gvB: filas de cada tramo. */
void ternary_avx512_attn_pv2(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                             size_t nA, size_t nB, size_t spA, const float *vA, const float *vB, size_t vs, size_t H,
                             float *e, size_t es, float *out, float *inv_out);
void ternary_avx512_attn_bwd2(size_t G, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs,
                              const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga,
                              const float *o, const float *vTA, const float *vTB, size_t ld, float it, float gam,
                              const int32_t *qr, const int32_t *krA, const int32_t *krB, size_t ks, float *gq,
                              float *gkA, float *gkB, float *gvA, float *gvB, size_t gs, float *p, float *ds,
                              size_t es, float *qf);


/* ---- Entrada uint8 (ternary_u8.h): mismos bits que el escalar (enteros exactos / un mul y un add por elemento) ---- */
/* sum_i w_i*(q_i - zp) con mask/sign de n bits (bits sobrantes en 0): igual que ternary_dot_u8_scalar. */
int32_t ternary_avx512_dot_u8(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp);
/* out[i] = sx * (float)(q[i] - zp). */
void ternary_avx512_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out);
/* gw[i] += g * xf[i] (un mul y un add separados, sin FMA). */
void ternary_avx512_outer_acc(float *gw, size_t n, float g, const float *xf);
/* gx[i] += gs donde w_i = +1, gx[i] -= gs donde w_i = -1 (bits de mask/sign desde el bit 0). */
void ternary_avx512_gradx_row(float *gx, size_t n, const uint64_t *wm, const uint64_t *ws, float gs);


/* ---- SwiGLU, elemento a elemento (modo: 0 silu, 1 sign, 2 ReLU2). Mismos bits que el escalar de ternary_swiglu.c ---- */
void ternary_avx512_swiglu_prod_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, float su, const float *silu_tab, int32_t D, float *h);
void ternary_avx512_swiglu_bwd_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float su,
                                 const float *silu_tab, const float *dsilu_tab, int32_t D, float *gu, float *ga);
void ternary_avx512_swiglu_fwd_u8(int mode, size_t H, const float *fa, const float *fu, const int32_t *zg, float *h);
void ternary_avx512_swiglu_bwd_u8(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float kg, float ku,
                                  float *gu, float *ga);


/* ---- Entropia cruzada y cabeza ---- */
/* Cuerpo de ternary_xent_row (ternary_loss.c) con la exponencial compartida y suma de 16 carriles en double. */
void ternary_avx512_xent_row(const float *l, size_t V, size_t t, size_t n, float *g, int *ok_out, double *loss_out);
/* sum (double)g[v]*(double)z[v] (use_sx = 0) o ((double)g[v]*sx)*(double)z[v] (use_sx = 1): ternary_canon_dot_gz[sx]. */
double ternary_avx512_dot_gz(const float *g, const int32_t *z, size_t n, int use_sx, double sx);
/* x[i] *= s (un mul por elemento). */
void ternary_avx512_scale_f32(float *x, size_t n, float s);


/* ---- Regla R2 de una fila de n pesos que empiezan en el bit 'base' de los planos mask/sign, con sus contadores cnt[0..n) (OJO: ya desplazado
 * por el llamador); umbrales y reinicio ya calculados (ternary_r2_thresholds). Mismos bits que ternary_r2_row_scalar. ---- */
/* Una fila de la normalizacion por elemento (mismos bits que ternary_elemnorm_apply_scalar). */
void ternary_avx512_elemnorm_row(uint8_t *cr, float *rmax, float *g, float *v, size_t W, float beta2, float omb, float bc, float eps);
void ternary_avx512_r2_row(uint64_t *mask, uint64_t *sign, int8_t *cnt, size_t base, const float *g, size_t n, float beta, float inc, int ti, int tin, int tout, int ri, int pre);


/* ---- Elementos sueltos: mismos bits que los bucles escalares que reemplazan ---- */
/* Cuantiza UNA fila a uint8 (igual que una fila de ternary_u8_quantize). */
void ternary_avx512_u8_quantize_row(const float *x, size_t n, uint8_t *q, float *scale_out, int32_t *zp_out);
void ternary_avx512_add_f32(float *out, const float *a, const float *b, size_t n);          /* out = a + b */
void ternary_avx512_acc_add2(float *dst, const float *a, const float *b, size_t n);         /* dst += (a + b) */
/* Softmax entero (ternary_softmax_forward/backward): suma de 16 carriles (orden canonico). out[n_valid..] lo pone el llamador. */
void ternary_avx512_softmax_fwd(const int32_t *scores, size_t n_valid, const float *table, size_t table_len, float *out);
void ternary_avx512_softmax_bwd(const float *y, const float *g, size_t n, float inv_temp, float *out_ds);
/* RoPE: 'heads' cabezas de ancho H con P pares; c/s = cos/sin en punto fijo de la posicion. */
void ternary_avx512_rope_rotate_int(const int32_t *z, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, int32_t *out);
void ternary_avx512_rope_unrotate_grad(float *grad, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, float inv);
/* out[i] = (z[i] + 2^(shift-1)) >> shift (aritmetico). */
void ternary_avx512_shift_round_i32(const int32_t *z, size_t n, int shift, int32_t *out);
/* out[i*stride] = (float)sign(z[i])  /  out[i*stride] = k * (float)z[i]. */
void ternary_avx512_sign_f32(const int32_t *z, size_t n, float *out, size_t stride);
void ternary_avx512_scale_i32_f32(const int32_t *z, size_t n, float k, float *out, size_t stride);

/* dst[i*stride] = src[i] (transposicion de K: una columna). */
void ternary_avx512_scatter_i32(const int32_t *src, size_t n, int32_t *dst, size_t stride);
/* Smeared keys, un vector de n enteros: out[i] = first ? in[i] : (in[i] + prev[i]) >> 1 (aritmetico) y despues prev[i] = in[i].
 * 'out' puede ser el mismo buffer que 'in'. Bit a bit igual a la version escalar de ternary_gqa.c. */
void ternary_avx512_smear_k(const int32_t *in, int32_t *out, int32_t *prev, size_t n, int first);
/* Backward de las smeared keys, EN SITIO sobre g (T filas de w floats): fila p = (p == 0 ? 1 : 0.5) * g[p] + (p + 1 < T ? 0.5 * g[p+1] : 0). */
/* Posiciones [p0, p1) de T; la fila p1 (si p1 < T) puede venir en next_of_last (copia hecha antes de repartir); NULL = leerla en sitio. */
void ternary_avx512_smear_grad_rows(float *g, size_t p0, size_t p1, size_t T, size_t w, const float *next_of_last);

/* out[i] = (float)w_i con w_i en {-1,0,+1} dado por los planos mask/sign (bit i). Los bits >= n no se leen como valores. */
void ternary_avx512_expand_trits(const uint64_t *mask, const uint64_t *sign, size_t n, float *out);

/* mask/sign bit (bit0 + i) = (z[i] != 0) / (z[i] < 0) para i < n; el resto de los bits queda como estaba. */
void ternary_avx512_pack_sign_i32(const int32_t *z, size_t n, uint64_t *mask, uint64_t *sign, size_t bit0);

/* Primer indice i en [from, n) con ids[i] en [lo, hi) (n si no hay; lo >= hi = rango vacio). Es la busqueda de marcadores
 * especiales de ternary_targets (16 ids por comparacion). Entero: identico al bucle escalar. */
size_t ternary_avx512_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi);

#endif /* TERNARY_AVX512_H */
