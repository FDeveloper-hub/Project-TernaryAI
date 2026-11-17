#ifndef TERNARYAI_INFER_H
#define TERNARYAI_INFER_H
/*
 * ternary_infer: motor de inferencia para modelos TernaryAI (carpeta con model.ternary + config.json [+ tokenizer.json]).
 *
 *  - C11 puro, sin dependencias (el JSON y el tokenizador BPE vienen en vendor/).
 *  - Decodificacion con cache de claves/valores: cada token nuevo calcula UNA posicion.
 *  - Dos nucleos con los mismos bits: escalar (portable) y AVX-512 (VNNI, o BW si no hay VNNI), con despacho en
 *    ejecucion. TERNARYAI_NO_AVX512=1 o ternaryai_infer_set_backend() fuerzan el escalar.
 *  - Un solo hilo. Los objetos no son seguros entre hilos (cada sesion en su hilo; el modelo se comparte solo lectura,
 *    salvo la tabla de RoPE, que crece perezosamente: crear/avanzar sesiones desde un solo hilo).
 *
 * Convencion: las funciones devuelven 0 o un valor >= 0 en exito y un codigo TERNARYAI_INFER_E_* negativo si fallan.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#if defined(__GNUC__)
#  pragma GCC visibility push(default)   /* la libreria compila con -fvisibility=hidden: esta API si se exporta */
#endif

#define TERNARYAI_INFER_ABI 1

enum {
    TERNARYAI_INFER_OK = 0,
    TERNARYAI_INFER_E_ARG = -1,        /* argumento invalido */
    TERNARYAI_INFER_E_NOMEM = -2,
    TERNARYAI_INFER_E_CTX_FULL = -3,   /* contexto lleno con politica TERNARYAI_INFER_CTX_ERROR */
    TERNARYAI_INFER_E_UNSUPPORTED = -4,
    TERNARYAI_INFER_E_NOTOK = -5,      /* el modelo no trae tokenizer.json */
    TERNARYAI_INFER_E_IO = -6,
    TERNARYAI_INFER_E_STATE = -7,      /* p. ej. pedir logits sin haber procesado ningun token */
    TERNARYAI_INFER_E_RANGE = -8       /* id de token fuera del vocabulario */
};

/* ---------------------------------------------------------------- nucleo (backend) */
typedef enum { TERNARYAI_INFER_BACKEND_AUTO = 0, TERNARYAI_INFER_BACKEND_SCALAR = 1, TERNARYAI_INFER_BACKEND_AVX512 = 2, TERNARYAI_INFER_BACKEND_NEON = 3 } TernaryInferBackend;
/* Fija el nucleo global. Devuelve el backend activo (TERNARYAI_INFER_BACKEND_SCALAR/AVX512) o TERNARYAI_INFER_E_UNSUPPORTED si se pidio AVX-512
 * y la CPU no lo tiene. Llamarlo antes de crear sesiones (o con ninguna sesion en curso). */
int ternaryai_infer_set_backend(TernaryInferBackend b);
int ternaryai_infer_get_backend(void);
/* Hilos del motor: reparte cada producto matriz x vector (y la cabeza de salida) por bloques de filas. n <= 1: un solo hilo. Se elige a mano;
 * el resultado es el mismo bit a bit con cualquier numero (aritmetica entera). En libternaryai es el mismo ajuste que ternaryai_set_threads.
 * Llamarlo sin generacion en curso. */
void ternaryai_infer_set_threads(int n);
int ternaryai_infer_get_threads(void);
const char *ternaryai_infer_backend_name(void);
int ternaryai_infer_cpu_has_avx512(void);
int ternaryai_infer_cpu_has_neon(void);     /* ARM con NEON (el procesador lo declara y el binario se compilo con NEON) */

/* ---------------------------------------------------------------- modelo */
typedef struct TernaryInferModel TernaryInferModel;

typedef struct {
    int vocab, dim, seq_len;          /* seq_len = contexto con el que se entreno */
    int n_residual, n_attn, n_ffn;
    int n_heads, n_kv_heads, head_dim, hidden;   /* de la primera capa GQA / feed-forward (0 si no hay) */
    int gate_mode;                    /* 0 silu, 1 sign, 2 ReLU^2 */
    int smear, window, sinks;         /* de la primera capa GQA */
    int has_tokenizer;
    size_t weight_bytes;              /* memoria de pesos en uso: 2 bits por peso (una sola copia) + sumas por fila + normas */
    uint64_t fingerprint;             /* hash de pesos y arquitectura (cache guardada ligada al modelo) */
} TernaryInferModelInfo;

/* Orden de los pesos en memoria / en un .ternam: uno por familia de nucleo. */
#define TERNARYAI_INFER_LAYOUT_ROW 0     /* escalar: fila a fila */
#define TERNARYAI_INFER_LAYOUT_AVX 1     /* AVX-512: bloques de 16 filas x 16 columnas */
#define TERNARYAI_INFER_LAYOUT_NEON 2    /* NEON: fila a fila, grupos de 64 columnas en 16 bytes */
#define TERNARYAI_INFER_LAYOUT_AUTO (-1) /* el del nucleo activo */

/* 'path' = carpeta con model.ternary y config.json (como la crea Model.save / Model.export) o un archivo .ternam (modelo autocontenido). */
TernaryInferModel *ternaryai_infer_model_load(const char *path, char *err, size_t errcap);
/* Escribe 'out' (.ternam) a partir de una carpeta de modelo o de otro .ternam, con los pesos en el orden 'layout' (TERNARYAI_INFER_LAYOUT_*).
 * Devuelve 0 o un codigo TERNARYAI_INFER_E_* negativo (el mensaje queda en err). */
/* Se suma a 'layout' en export_ternam: no guarda el tokenizador (para librerias fusionadas sin encode/decode). */
#define TERNARYAI_INFER_EXPORT_NO_TOKENIZER 0x100
/* Solo en librerias FUSIONADAS (tn.export_fused): el modelo incrustado (NULL si no se pudo preparar). No existen model_load ni export_ternam. */
TernaryInferModel *ternaryai_infer_fused_model(void);
/* Escribe el C estatico del modelo fusionado (lo usa tn.export_fused): 'ternam' es un .ternam; with_tok=0 no enlaza el tokenizador. */
int ternaryai_infer_export_fused_c(const char *ternam, const char *out_c, int with_tok, char *err, size_t errcap);
int ternaryai_infer_export_ternam(const char *src, const char *out, int layout, char *err, size_t errcap);
void ternaryai_infer_model_free(TernaryInferModel *m);
void ternaryai_infer_model_info(const TernaryInferModel *m, TernaryInferModelInfo *out);

/* ---------------------------------------------------------------- sesion */
typedef enum {
    TERNARYAI_INFER_CTX_ERROR = 0,      /* contexto lleno -> TERNARYAI_INFER_E_CTX_FULL */
    TERNARYAI_INFER_CTX_TRUNCATE = 1,   /* conserva los ultimos keep_tokens y los reprocesa (posiciones desde 0: misma semantica del entrenamiento) */
    TERNARYAI_INFER_CTX_SLIDE = 2       /* descarta los mas viejos tras los sinks; las posiciones siguen creciendo (RoPE extrapolada, experimental) */
} TernaryInferCtxPolicy;

typedef struct {
    int max_context;       /* ranuras de la cache (por defecto seq_len del modelo; con TERNARYAI_INFER_CTX_SLIDE puede ser menor o mayor) */
    int attn_window;       /* 0 = atender a toda la cache; W > 0 = a los sinks y las ultimas W claves */
    int sinks;             /* posiciones iniciales siempre visibles / protegidas */
    int ctx_policy;        /* TernaryInferCtxPolicy */
    int keep_tokens;       /* TERNARYAI_INFER_CTX_TRUNCATE: cuantos tokens se conservan (0 = max_context / 2) */
    int record_attn;       /* guarda las probabilidades de atencion del ultimo token (ternaryai_infer_attn_probs) */
    int max_history;       /* tokens recordados para penalizaciones y paradas (0 = ilimitado) */
} TernaryInferSessionOpts;

typedef struct TernaryInferSession TernaryInferSession;

void ternaryai_infer_session_opts_default(const TernaryInferModel *m, TernaryInferSessionOpts *o);
TernaryInferSession *ternaryai_infer_session_new(const TernaryInferModel *m, const TernaryInferSessionOpts *opts);   /* opts NULL = valores por defecto */
TernaryInferSession *ternaryai_infer_session_fork(const TernaryInferSession *s);                           /* copia independiente (cache, historia, semilla) */
void ternaryai_infer_session_free(TernaryInferSession *s);
void ternaryai_infer_session_reset(TernaryInferSession *s);                                      /* vacia el contexto (conserva opciones y semilla) */
int ternaryai_infer_session_len(const TernaryInferSession *s);                                   /* claves en la cache */
int ternaryai_infer_session_next_position(const TernaryInferSession *s);                         /* posicion absoluta del proximo token */
int ternaryai_infer_session_history(const TernaryInferSession *s, const int32_t **ids);          /* historia de tokens (puntero valido hasta el proximo cambio) */
const TernaryInferModel *ternaryai_infer_session_model(const TernaryInferSession *s);
void ternaryai_infer_session_seed(TernaryInferSession *s, uint64_t seed);

/* Guarda / restaura la cache de una sesion (por ejemplo, un prompt de sistema ya procesado). */
int ternaryai_infer_session_save(const TernaryInferSession *s, const char *path);
TernaryInferSession *ternaryai_infer_session_load(const TernaryInferModel *m, const char *path, const TernaryInferSessionOpts *opts, char *err, size_t errcap);

/* ---------------------------------------------------------------- procesar tokens */
/* Procesa n tokens (prefill). Al terminar quedan listos los logits del ULTIMO token (ternaryai_infer_argmax / ternaryai_infer_logits /
 * ternaryai_infer_sample). Con n = 0 no hace nada. */
int ternaryai_infer_feed(TernaryInferSession *s, const int32_t *ids, int n);
int ternaryai_infer_feed1(TernaryInferSession *s, int32_t id);

/* ---------------------------------------------------------------- salida del ultimo token procesado */
/* Voraz: token de mayor logit (el primero en empate), calculado en enteros. TERNARYAI_INFER_E_STATE si no hay logits. */
int ternaryai_infer_argmax(const TernaryInferSession *s, int32_t *id_out);
/* Logits float del ultimo token (vocab valores), identicos a los de predict de la libreria. */
int ternaryai_infer_logits(const TernaryInferSession *s, float *out);
/* Top-n por probabilidad (softmax a temperatura 1), de mayor a menor. */
int ternaryai_infer_top_probs(const TernaryInferSession *s, int n, int32_t *ids_out, float *probs_out);

/* ---------------------------------------------------------------- muestreo */
typedef struct {
    float temperature;       /* <= 0: voraz */
    int top_k;               /* 0 = sin limite */
    float top_p;             /* >= 1 o <= 0: sin limite */
    float min_p;             /* 0 = apagado: descarta p < min_p * p_max */
    float repeat_penalty;    /* 1 = apagado; > 1 penaliza tokens ya vistos (divide logits > 0, multiplica los < 0) */
    float freq_penalty;      /* 0 = apagado: logit -= freq_penalty * conteo */
    float presence_penalty;  /* 0 = apagado: logit -= presence_penalty si el token ya aparecio */
    int penalty_last_n;      /* ventana de la historia para las penalizaciones (0 = toda la historia) */
    const float *logit_bias;     /* vocab valores sumados a los logits, o NULL */
    const uint8_t *allowed;      /* vocab bytes (1 = permitido), o NULL */
    const int32_t *banned;       /* tokens prohibidos */
    int n_banned;
} TernaryInferSampler;
void ternaryai_infer_sampler_default(TernaryInferSampler *p);    /* voraz, todo apagado */
/* Elige el proximo token segun p (usa el generador de la sesion: reproducible con ternaryai_infer_session_seed). */
int ternaryai_infer_sample(TernaryInferSession *s, const TernaryInferSampler *p, int32_t *id_out);

/* ---------------------------------------------------------------- generacion */
typedef enum { TERNARYAI_INFER_STOP_NONE = 0, TERNARYAI_INFER_STOP_MAX_TOKENS, TERNARYAI_INFER_STOP_TOKEN, TERNARYAI_INFER_STOP_SEQUENCE, TERNARYAI_INFER_STOP_CALLBACK, TERNARYAI_INFER_STOP_ERROR } TernaryInferStopReason;

typedef struct TernaryInferGenParams {
    int max_new_tokens;
    TernaryInferSampler sampler;
    const int32_t *stop_ids; int n_stop_ids;               /* tokens de parada (p. ej. <|end|>) */
    const int32_t *stop_seq; const int *stop_seq_len; int n_stop_seq;   /* secuencias de parada: stop_seq = concatenacion */
    int include_stop;                                      /* 1: el token/secuencia de parada tambien se entrega */
    int (*on_token)(int32_t id, void *user);               /* streaming: != 0 detiene la generacion */
    void *user;
    /* Mascara por paso: rellena allowed_out[vocab] (1 = permitido). Devuelve != 0 si la usa. Opcional. */
    int (*mask_fn)(void *user, const TernaryInferSession *s, uint8_t *allowed_out);
    /* Presupuesto de razonamiento. Si think_budget > 0 y ambos ids >= 0: tras think_open_id se cuentan los tokens generados; al llegar a
     * think_budget sin haber cerrado, el siguiente token NO se muestrea: se fuerza think_close_id (entra al contexto y a la salida).
     * think_inside = 1: la generacion empieza ya dentro del bloque (el prompt termina en think_open_id). Con think_budget = 0 no hace nada. */
    int think_open_id, think_close_id, think_budget, think_inside;
} TernaryInferGenParams;
void ternaryai_infer_gen_params_default(TernaryInferGenParams *p);

/* Genera hasta max_new_tokens a partir del estado actual de la sesion (que debe tener logits: llamar antes a ternaryai_infer_feed).
 * Escribe los ids en out (capacidad cap). Devuelve cuantos escribio (>= 0) o un codigo de error; why recibe el motivo. */
int ternaryai_infer_generate(TernaryInferSession *s, const TernaryInferGenParams *p, int32_t *out, int cap, TernaryInferStopReason *why);

/* Varias sesiones a la vez, un paso por ronda; cada una se detiene por su cuenta. outs[i] tiene capacidad caps[i];
 * counts[i] y whys[i] reciben el resultado. Mismos parametros para todas (el generador de cada sesion es suyo). */
int ternaryai_infer_generate_batch(TernaryInferSession **ss, int n, const TernaryInferGenParams *p, int32_t **outs, const int *caps, int *counts, TernaryInferStopReason *whys);

/* n candidatas desde el mismo estado: forkea la sesion n veces (semillas semilla+i). outs/caps/counts como arriba. */
int ternaryai_infer_generate_n(const TernaryInferSession *s, int n, const TernaryInferGenParams *p, int32_t **outs, const int *caps, int *counts, TernaryInferStopReason *whys);

/* ---------------------------------------------------------------- evaluacion y diagnostico */
/* Procesa ids y devuelve log p(ids[i] | contexto anterior). Si la sesion ya tenia logits, logprob[0] usa esos; si no,
 * logprob[0] = NaN. Devuelve el numero de tokens puntuados (n o n-1). ppl_out (opcional) recibe exp(-media). */
int ternaryai_infer_score(TernaryInferSession *s, const int32_t *ids, int n, float *logprob_out, double *ppl_out);

/* Probabilidades de atencion del ULTIMO token (requiere record_attn): out[j] para las claves visibles, en orden de
 * cache; *n_out = cuantas. attn_layer = indice de capa GQA (0..n_attn-1). */
int ternaryai_infer_attn_probs(const TernaryInferSession *s, int attn_layer, int head, float *out, int cap, int *n_out);

/* ---------------------------------------------------------------- texto (si hay tokenizer.json) */
int ternaryai_infer_encode(const TernaryInferModel *m, const char *text, int parse_special, int32_t *ids, int cap, int *n_out);
int ternaryai_infer_decode(const TernaryInferModel *m, const int32_t *ids, int n, int skip_special, char *out, int cap, int *len_out);
int ternaryai_infer_special_id(const TernaryInferModel *m, const char *text);    /* id del token especial, o -1 */
/* Conversacion -> ids, FORMATO ANTIGUO (sin separadores): cada mensaje "<|rol|>texto"; con add_assistant_prompt agrega "<|assistant|>"
 * al final. roles[i] es "user" o "assistant" (o "system"). Se conserva para modelos entrenados con ese formato; para modelos entrenados con
 * ternaryai_dataset_* (o con formato propio) usar ternaryai_infer_chat_encode_fmt / ternaryai_infer_chat_reply_ex. */
int ternaryai_infer_chat_encode(const TernaryInferModel *m, const char *const *roles, const char *const *texts, int n_msgs,
                   int add_assistant_prompt, int32_t *ids, int cap, int *n_out);
/* Respuesta del asistente a una conversacion: procesa el prompt en la sesion (que se reinicia) y genera hasta
 * <|end|>. Escribe texto UTF-8 en out. Devuelve bytes escritos o un error. */
int ternaryai_infer_chat_reply(TernaryInferSession *s, const char *const *roles, const char *const *texts, int n_msgs,
                  const TernaryInferGenParams *p, char *out, int cap, TernaryInferStopReason *why);

/* ---- conversacion configurable (mismo formato que el cargador de datasets: "<|rol|>\ntexto\n", turno del asistente "<|assistant|>\n") ---- */
typedef struct TernaryInferChatOpts {
    const char *after_role;   /* texto tras el marcador de rol; NULL = "\n" (como el dataset). "" = sin separadores */
    const char *after_msg;    /* texto tras cada mensaje; NULL = "\n" */
    int stop_on_user;         /* 1 (defecto): <|user|> tambien detiene la respuesta (modelos entrenados con turnos encadenados) */
    int force_think;          /* 1: el prompt termina en <|think|> (el modelo empieza pensando) */
    int think_budget;         /* > 0: maximo de tokens de razonamiento; al llegar se fuerza <|/think|>. 0 = sin limite */
    int show_think;           /* 1: el texto devuelto conserva el bloque <|think|>...<|/think|> en su sitio; 0 (defecto): se quita del texto */
} TernaryInferChatOpts;
void ternaryai_infer_chat_opts_default(TernaryInferChatOpts *o);

/* Igual que chat_encode pero con los separadores de o (NULL = valores por defecto de arriba). Cada mensaje se codifica como un solo tramo
 * "<after_role>texto<after_msg>" (igual que el cargador de datasets; el BPE no es independiente por trozos). El prompt del asistente es solo
 * el marcador <|assistant|>: el modelo genera el separador por si mismo (el dataset lo entrena como parte del turno). Con o->force_think
 * se agregan ademas after_role y <|think|>. */
int ternaryai_infer_chat_encode_fmt(const TernaryInferModel *m, const TernaryInferChatOpts *o, const char *const *roles, const char *const *texts,
                   int n_msgs, int add_assistant_prompt, int32_t *ids, int cap, int *n_out);

/* Separa los ids generados en respuesta y razonamiento: lo que esta entre <|think|> y <|/think|> (sin las etiquetas) va a think_out
 * (varios bloques se concatenan), el resto a answer_out (sin las etiquetas). Un bloque sin cerrar llega hasta el final. think_inside = 1: los
 * ids empiezan dentro de un bloque. Cualquiera de las salidas puede ser NULL; capacidades >= n bastan. Devuelve 0 o un error
 * (E_UNSUPPORTED si el tokenizador no tiene las etiquetas). */
int ternaryai_infer_think_split(const TernaryInferModel *m, const int32_t *ids, int n, int think_inside,
                   int32_t *answer_out, int *n_answer, int32_t *think_out, int *n_think);

/* Respuesta del asistente con las opciones de o (NULL = por defecto). Procesa el prompt en la sesion (que se reinicia), genera hasta <|end|>
 * (y <|user|> si stop_on_user) y escribe en out el texto de la respuesta y en think_out (opcional) el razonamiento. Devuelve los bytes escritos
 * en out o un error. why recibe el motivo de parada. */
int ternaryai_infer_chat_reply_ex(TernaryInferSession *s, const TernaryInferChatOpts *o, const char *const *roles, const char *const *texts, int n_msgs,
                  const TernaryInferGenParams *p, char *out, int cap, char *think_out, int think_cap, TernaryInferStopReason *why);

/* ---------------------------------------------------------------- medicion */
/* Perfilador por ambitos (apagado por defecto; con el encendido cada ambito cuesta ~20 ns). */
void ternaryai_infer_prof_enable(int on);
void ternaryai_infer_prof_reset(void);
int ternaryai_infer_prof_report(char *buf, size_t cap);     /* texto legible; devuelve bytes escritos */

/* Reloj monotono en nanosegundos (para las herramientas de medicion). */
uint64_t ternaryai_infer_now_ns(void);

#if defined(__GNUC__)
#  pragma GCC visibility pop
#endif
#ifdef __cplusplus
}
#endif
#endif /* TERNARYAI_INFER_H */
