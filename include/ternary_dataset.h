#ifndef TERNARY_DATASET_H
#define TERNARY_DATASET_H
#include <stddef.h>
#include <stdint.h>
/*
 * Carga archivos de entrenamiento (texto plano, JSON y JSONL) y los deja listos como
 * texto plano para el tokenizador (ver ternary_tokenizer.h):
 *
 *  - TEXT_LINES / TEXT_WHOLE: archivo .txt de texto plano. LINES trata
 *    cada linea no vacia como un ejemplo independiente; WHOLE trata el
 *    archivo entero como un solo ejemplo. Configurable por llamada (no
 *    hay una unica convencion fija).
 *  - JSON: un arreglo JSON de registros. Todo el archivo se lee de una
 *    sola vez -- para datasets grandes conviene JSONL.
 *  - JSONL: un registro por linea -- procesa linea a linea, mas liviano
 *    en memoria para datasets grandes.
 *
 * En JSON/JSONL el formato se detecta POR REGISTRO segun la clave que
 * traiga (si trae varias, gana la primera de esta lista), asi que un
 * mismo archivo puede mezclar formatos:
 *   "chat"           formato propio: [{"user": "..."}, {"model": "..."}, ...]
 *                    un turno = un objeto con UNA clave (el rol) y su texto.
 *                    Roles: user, model, system, console, terminal, archive,
 *                    log, result, function_call, tool (sin distinguir mayusculas); otro -> error.
 *   "messages"       ChatML/OpenAI: [{"role": "...", "content": "..."}, ...]
 *   "conversations"  ShareGPT: [{"from": "human"|"gpt"|..., "value": "..."}, ...]
 *                    (tambien acepta elementos {"role", "content"})
 *   "prompt" + "completion"   cadenas, o listas de mensajes {"role","content"}
 *   "instruction" [+ "input"] + "output"   Alpaca: el turno user es la
 *                    instruccion y, si hay input no vacio, un parrafo aparte
 *   "text"           texto plano: el ejemplo es el texto tal cual, SIN
 *                    marcadores de turno ni <|end|>
 * Los demas campos del registro (id, source, ...) se ignoran.
 * Roles (sin distinguir mayusculas): user/human -> user; assistant/gpt/model ->
 * assistant; system, console, terminal, archive, log, result, function_call y tool quedan igual.
 * En messages/conversations un rol desconocido se conserva tal cual.
 *
 * Cada conversacion se renderiza a texto con un formato de turnos fijo:
 *   "<|" + role + "|>\n" + content + "\n"   (uno por mensaje)
 *   "<|end|>\n"                              (al final de la conversacion)
 * Son marcadores de texto plano, no tokens especiales reservados (el
 * tokenizador BPE los aprendera como cualquier otro patron repetido del
 * corpus) -- mas simple, y sigue funcionando si mas adelante se agregan
 * tokens especiales de verdad sin cambiar este formato de renderizado.
 *
 * Todos los ejemplos (uno por linea/objeto/conversacion segun el
 * formato) se devuelven concatenados en un solo buffer, cada uno
 * terminado con un byte NUL (0x00) como separador -- mismo patron de
 * medicion "llamar primero para medir" que el resto de la API plana.
 */

enum {
    TERNARY_DATASET_TEXT_LINES = 0,
    TERNARY_DATASET_TEXT_WHOLE = 1,
    TERNARY_DATASET_JSON       = 2,
    TERNARY_DATASET_JSONL      = 3,
};

/* strict=1: un registro/linea malformado detiene la carga con error.
 * strict=0: se saltea el registro malformado y se sigue; *out_skipped
 * (puede ser NULL) cuenta cuantos se saltearon. strict no aplica a
 * TEXT_* (ahi no hay nada que pueda estar "malformado"). */
int ternary_dataset_load(const char *path, int format, int strict,
                          uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                          size_t *out_skipped, char *err, size_t err_cap);

/* Igual que ternary_dataset_load, con opciones (flags, combinables con |):
 *  TERNARY_DATASET_REASONING_KEEP (0, el de siempre): las etiquetas <think>...</think> / <thinking>...</thinking> del texto del
 *    asistente se normalizan a <|think|>...<|/think|> y un campo 'reasoning_content' / 'reasoning' / 'thinking' del mensaje se escribe
 *    antes de la respuesta ("<|think|>\n<razonamiento>\n<|/think|>\n\n<respuesta>").
 *  TERNARY_DATASET_REASONING_DROP: se quita todo razonamiento (campos y bloques entre etiquetas) para entrenar modelos que no razonan.
 * Las etiquetas deben estar balanceadas; si no, el registro es invalido (error en strict, saltado si no). Texto plano y
 * registros 'text' no se tocan. */
enum {
    TERNARY_DATASET_REASONING_KEEP = 0,
    TERNARY_DATASET_REASONING_DROP = 1,
};
int ternary_dataset_load_ex(const char *path, int format, int strict, int flags,
                             uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                             size_t *out_skipped, char *err, size_t err_cap);

#endif /* TERNARY_DATASET_H */
