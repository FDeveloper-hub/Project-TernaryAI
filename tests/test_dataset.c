/*
 * Verifica ternary_dataset_load para los 3 formatos:
 *  - texto plano (lineas vs bloque completo)
 *  - JSON (arreglo de conversaciones)
 *  - JSONL (una conversacion por linea)
 * incluyendo: renderizado de turnos correcto, UTF-8 con escapes \u,
 * campos ignorados en el esquema, modo strict vs no-strict con
 * registros malformados, y que un separador NUL entre ejemplos alcance
 * para reconstruirlos del lado de quien llama (aqui mismo).
 */
#include "tmp_path.h"
#include "ternary_dataset.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char err[512];

static int write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite(content, 1, strlen(content), f);
    fclose(f);
    return 1;
}

/* Carga con el patron medir->reservar->llenar y devuelve el buffer
 * (el llamador hace free). */
static uint8_t *load_alloc(const char *path, int format, int strict, size_t *out_len, size_t *out_count,
                            size_t *out_skipped) {
    int ok = ternary_dataset_load(path, format, strict, NULL, 0, out_len, out_count, out_skipped, err, sizeof(err));
    if (!ok) return NULL;
    uint8_t *buf = malloc(*out_len ? *out_len : 1);
    ok = ternary_dataset_load(path, format, strict, buf, *out_len, out_len, out_count, out_skipped, err, sizeof(err));
    assert(ok);
    return buf;
}

/* Devuelve el ejemplo N-esimo (separado por NUL) como string C recien
 * asignado (el llamador hace free), o NULL si no hay tantos. */
static char *nth_example(const uint8_t *buf, size_t len, size_t count, size_t n) {
    if (n >= count) return NULL;
    size_t i = 0, seen = 0, start = 0;
    for (; i < len; i++) {
        if (buf[i] == 0) {
            if (seen == n) {
                size_t exlen = i - start;
                char *out = malloc(exlen + 1);
                memcpy(out, buf + start, exlen);
                out[exlen] = '\0';
                return out;
            }
            seen++;
            start = i + 1;
        }
    }
    return NULL;
}

static void test_texto_lineas_vs_bloque(void) {
    const char *path = tmp_path("test_dataset_texto.txt");
    write_file(path, "primera linea\n\nsegunda linea\r\nultima sin salto");

    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_TEXT_LINES, 1, &len, &count, &skipped);
    assert(buf != NULL);
    printf("[texto lineas] %zu ejemplos (se esperaban 3, la linea vacia se saltea)\n", count);
    assert(count == 3);
    char *e0 = nth_example(buf, len, count, 0);
    char *e1 = nth_example(buf, len, count, 1);
    char *e2 = nth_example(buf, len, count, 2);
    assert(strcmp(e0, "primera linea") == 0);
    assert(strcmp(e1, "segunda linea") == 0);  /* \r del CRLF recortado */
    assert(strcmp(e2, "ultima sin salto") == 0);
    free(e0); free(e1); free(e2); free(buf);

    buf = load_alloc(path, TERNARY_DATASET_TEXT_WHOLE, 1, &len, &count, &skipped);
    assert(count == 1);
    char *whole = nth_example(buf, len, count, 0);
    printf("[texto bloque] 1 ejemplo de %zu bytes\n", strlen(whole));
    assert(strcmp(whole, "primera linea\n\nsegunda linea\r\nultima sin salto") == 0);
    free(whole); free(buf);
    remove(path);
}

static void test_jsonl_renderiza_turnos(void) {
    const char *path = tmp_path("test_dataset.jsonl");
    write_file(path,
        "{\"messages\": [{\"role\": \"user\", \"content\": \"hola\"}, {\"role\": \"assistant\", \"content\": \"hola, como estas?\"}]}\n"
        "{\"id\": 42, \"messages\": [{\"role\": \"user\", \"content\": \"chau\"}], \"extra\": {\"a\": [1,2,3], \"b\": null}}\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    assert(buf != NULL);
    assert(count == 2);
    char *e0 = nth_example(buf, len, count, 0);
    const char *expected0 = "<|user|>\nhola\n<|assistant|>\nhola, como estas?\n<|end|>\n";
    printf("[jsonl turnos] ejemplo 0:\n%s", e0);
    assert(strcmp(e0, expected0) == 0);
    char *e1 = nth_example(buf, len, count, 1);
    const char *expected1 = "<|user|>\nchau\n<|end|>\n";
    assert(strcmp(e1, expected1) == 0);  /* campos 'id'/'extra' ignorados correctamente */
    printf("[jsonl turnos] campos desconocidos ('id', 'extra' anidado) ignorados OK\n");
    free(e0); free(e1); free(buf);
    remove(path);
}

static void test_json_arreglo_y_utf8(void) {
    const char *path = tmp_path("test_dataset.json");
    /* \u00f1 = 'n' con tilde, \u00e9 = 'e' con acento */
    write_file(path,
        "[\n"
        "  {\"messages\": [{\"role\": \"user\", \"content\": \"a\\u00f1o\"}, {\"role\": \"assistant\", \"content\": \"caf\\u00e9\"}]},\n"
        "  {\"messages\": [{\"role\": \"system\", \"content\": \"eres util\"}, {\"role\": \"user\", \"content\": \"hola\"}, {\"role\": \"assistant\", \"content\": \"hola!\"}]}\n"
        "]\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSON, 1, &len, &count, &skipped);
    assert(buf != NULL);
    assert(count == 2);
    char *e0 = nth_example(buf, len, count, 0);
    printf("[json utf8] ejemplo 0:\n%s", e0);
    assert(strstr(e0, "a\xc3\xb1o") != NULL);   /* "año" en UTF-8 */
    assert(strstr(e0, "caf\xc3\xa9") != NULL);  /* "café" en UTF-8 */
    char *e1 = nth_example(buf, len, count, 1);
    assert(strstr(e1, "<|system|>\neres util\n") != NULL);
    assert(strstr(e1, "<|user|>\nhola\n") != NULL);
    assert(strstr(e1, "<|assistant|>\nhola!\n") != NULL);
    printf("[json utf8] 3 roles distintos (system/user/assistant) renderizados OK\n");
    free(e0); free(e1); free(buf);
    remove(path);
}

static void test_strict_vs_no_strict(void) {
    const char *path = tmp_path("test_dataset_malformado.jsonl");
    write_file(path,
        "{\"messages\": [{\"role\": \"user\", \"content\": \"bien 1\"}]}\n"
        "{\"messages\": [{\"role\": \"user\"}]}\n"                                   /* falta 'content' */
        "{\"sin_messages\": true}\n"                                                 /* falta 'messages' */
        "{\"messages\": []}\n"                                                       /* 'messages' vacio */
        "{\"messages\": [{\"role\": \"user\", \"content\": \"bien 2\"}]}\n");

    size_t len, count, skipped;
    uint8_t *buf = ternary_dataset_load(path, TERNARY_DATASET_JSONL, 1 /*strict*/,
                                         NULL, 0, &len, &count, &skipped, err, sizeof(err)) ? (uint8_t *)1 : NULL;
    assert(buf == NULL);
    printf("[strict] con strict=1 y una linea malformada, falla como se esperaba: %s\n", err);

    buf = load_alloc(path, TERNARY_DATASET_JSONL, 0 /*no strict*/, &len, &count, &skipped);
    assert(buf != NULL);
    printf("[no strict] %zu ejemplos validos, %zu saltados (se esperan 2 y 3)\n", count, skipped);
    assert(count == 2);
    assert(skipped == 3);
    char *e0 = nth_example(buf, len, count, 0);
    char *e1 = nth_example(buf, len, count, 1);
    assert(strstr(e0, "bien 1") != NULL);
    assert(strstr(e1, "bien 2") != NULL);
    free(e0); free(e1); free(buf);
    remove(path);
}

static void test_json_array_no_strict_saltea_elemento(void) {
    /* Mismo caso pero en un arreglo JSON (no jsonl): un elemento
     * sintacticamente valido pero sin 'messages' debe poder saltearse
     * sin romper el parseo de los elementos siguientes. */
    const char *path = tmp_path("test_dataset_arr_malformado.json");
    write_file(path,
        "[{\"messages\": [{\"role\": \"user\", \"content\": \"ok1\"}]}, "
        "{\"otra_cosa\": [1, 2, {\"x\": \"y\"}]}, "
        "{\"messages\": [{\"role\": \"user\", \"content\": \"ok2\"}]}]");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSON, 0, &len, &count, &skipped);
    assert(buf != NULL);
    printf("[json arreglo no strict] %zu ejemplos, %zu saltados (se esperan 2 y 1)\n", count, skipped);
    assert(count == 2 && skipped == 1);
    char *e0 = nth_example(buf, len, count, 0);
    char *e1 = nth_example(buf, len, count, 1);
    assert(strstr(e0, "ok1") != NULL);
    assert(strstr(e1, "ok2") != NULL);
    free(e0); free(e1); free(buf);
    remove(path);
}


/* ---- formato propio "chat" -------------------------------------------------------------------------------------- */
static void test_chat_propio(void) {
    const char *path = tmp_path("test_dataset_chat.jsonl");
    write_file(path,
        "{\"chat\": [{\"user\": \"Hola\"},{\"model\": \"Hola\"}]} \n"
        "\n"
        "{\"chat\":[{\"user\":\"Hola\"},{\"model\":\"Hola. \\u00bfEn qu\\u00e9 puedo ayudarte?\"}, {\"user\":\"Quiero hablar un rato.\"}, "
        "{\"model\":\"Claro. \\u00bfSobre qu\\u00e9 te gustar\\u00eda hablar?\"},{\"user\":\"Sobre pel\\u00edculas.\"}, "
        "{\"model\":\"Perfecto. Podemos hablar de pel\\u00edculas que te gusten, g\\u00e9neros o personajes.\"}], \"id\": 7}\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    assert(buf && count == 2 && skipped == 0);
    char *e0 = nth_example(buf, len, count, 0), *e1 = nth_example(buf, len, count, 1);
    assert(strcmp(e0, "<|user|>\nHola\n<|assistant|>\nHola\n<|end|>\n") == 0);
    assert(strstr(e1, "<|user|>\nHola\n<|assistant|>\nHola. \xc2\xbf" "En qu\xc3\xa9 puedo ayudarte?\n<|user|>\nQuiero hablar un rato.\n") == e1);
    assert(strstr(e1, "<|user|>\nSobre pel\xc3\xad" "culas.\n<|assistant|>\nPerfecto.") != NULL);
    assert(strstr(e1, "<|end|>\n") == e1 + strlen(e1) - 8);
    printf("[chat] el formato propio (user/model) se renderiza igual que messages: OK\n");
    free(e0); free(e1); free(buf);

    /* mismo contenido en JSON (arreglo) y mezclado con 'messages' en el mismo archivo */
    write_file(path,
        "[{\"chat\": [{\"user\": \"a\"}, {\"model\": \"b\"}]}, "
        "{\"messages\": [{\"role\": \"user\", \"content\": \"a\"}, {\"role\": \"assistant\", \"content\": \"b\"}]}]");
    buf = load_alloc(path, TERNARY_DATASET_JSON, 1, &len, &count, &skipped);
    assert(buf && count == 2);
    e0 = nth_example(buf, len, count, 0); e1 = nth_example(buf, len, count, 1);
    assert(strcmp(e0, e1) == 0);
    printf("[chat] 'chat' y 'messages' en el mismo archivo dan el mismo texto: OK\n");
    free(e0); free(e1); free(buf);
    remove(path);
}

static void test_chat_roles(void) {
    const char *path = tmp_path("test_dataset_chat_roles.jsonl");
    write_file(path,
        "{\"chat\": [{\"system\": \"s\"}, {\"user\": \"u\"}, {\"model\": \"m\"}, {\"console\": \"c\"}, {\"terminal\": \"t\"}, "
        "{\"archive\": \"a\"}, {\"log\": \"l\"}, {\"result\": \"r\"}, {\"function_call\": \"f\"}, {\"Tool\": \"T\"}]}\n"
        "{\"chat\": [{\"User\": \"U\"}, {\"MODEL\": \"M\"}, {\"Console\": \"C\"}]}\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    assert(buf && count == 2);
    char *e0 = nth_example(buf, len, count, 0), *e1 = nth_example(buf, len, count, 1);
    assert(strcmp(e0, "<|system|>\ns\n<|user|>\nu\n<|assistant|>\nm\n<|console|>\nc\n<|terminal|>\nt\n"
                      "<|archive|>\na\n<|log|>\nl\n<|result|>\nr\n<|function_call|>\nf\n<|tool|>\nT\n<|end|>\n") == 0);
    assert(strcmp(e1, "<|user|>\nU\n<|assistant|>\nM\n<|console|>\nC\n<|end|>\n") == 0);   /* sin distinguir mayusculas */
    printf("[chat] 10 roles (system/user/model/console/terminal/archive/log/result/function_call/tool) y mayusculas: OK\n");
    free(e0); free(e1); free(buf);

    /* roles y estructuras invalidas: error en strict, saltados en no strict */
    const char *bad[] = {
        "{\"chat\": [{\"assistant\": \"x\"}]}\n",              /* 'assistant' no es un rol de 'chat' */
        "{\"chat\": [{\"human\": \"x\"}]}\n",
        "{\"chat\": [{\"narrator\": \"x\"}]}\n",
        "{\"chat\": [{\"user\": \"x\", \"model\": \"y\"}]}\n",     /* dos claves en el mismo turno */
        "{\"chat\": [{}]}\n",
        "{\"chat\": [{\"user\": 5}]}\n",                         /* texto no es cadena */
        "{\"chat\": [\"hola\"]}\n",                             /* turno no es objeto */
        "{\"chat\": []}\n",                                     /* vacio */
        "{\"chat\": \"hola\"}\n",                               /* no es lista */
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        write_file(path, bad[i]);
        assert(ternary_dataset_load(path, TERNARY_DATASET_JSONL, 1, NULL, 0, &len, &count, &skipped, err, sizeof(err)) == 0);
        buf = load_alloc(path, TERNARY_DATASET_JSONL, 0, &len, &count, &skipped);
        assert(buf && count == 0 && skipped == 1);
        free(buf);
    }
    printf("[chat] 9 casos invalidos: error en strict y 1 saltado en no strict: OK\n");
    remove(path);
}

static void test_messages_roles_y_text(void) {
    const char *path = tmp_path("test_dataset_roles.jsonl");
    write_file(path,
        "{\"messages\": [{\"role\": \"User\", \"content\": \"a\"}, {\"role\": \"model\", \"content\": \"b\"}, "
        "{\"role\": \"Tool\", \"content\": \"c\"}, {\"role\": \"Narrator\", \"content\": \"d\"}]}\n"                                  /* conocidos se normalizan; desconocidos quedan tal cual */
        "{\"text\": \"Un texto\\nsin turnos\", \"id\": 3}\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    assert(buf && count == 2);
    char *e0 = nth_example(buf, len, count, 0), *e1 = nth_example(buf, len, count, 1);
    assert(strcmp(e0, "<|user|>\na\n<|assistant|>\nb\n<|tool|>\nc\n<|Narrator|>\nd\n<|end|>\n") == 0);
    assert(strcmp(e1, "Un texto\nsin turnos") == 0);       /* text: tal cual, sin marcadores */
    printf("[messages/text] roles conocidos normalizados, desconocidos intactos, 'text' sin marcadores: OK\n");
    free(e0); free(e1); free(buf);
    write_file(path, "{\"text\": \"\"}\n{\"text\": 5}\n{\"otro\": 1}\n");
    buf = load_alloc(path, TERNARY_DATASET_JSONL, 0, &len, &count, &skipped);
    assert(buf && count == 0 && skipped == 3);
    free(buf);
    remove(path);
}

/* ---- formatos de Hugging Face ----------------------------------------------------------------------------------- */
static void test_formatos_hf(void) {
    const char *path = tmp_path("test_dataset_hf.jsonl");
    write_file(path,
        /* 0: ShareGPT */
        "{\"id\": \"x\", \"conversations\": [{\"from\": \"system\", \"value\": \"s\"}, {\"from\": \"human\", \"value\": \"hola\"}, "
        "{\"from\": \"GPT\", \"value\": \"qu\\u00e9 tal\"}, {\"from\": \"function_call\", \"value\": \"f()\"}]}\n"
        /* 1: ShareGPT con role/content */
        "{\"conversations\": [{\"role\": \"user\", \"content\": \"a\"}, {\"role\": \"assistant\", \"content\": \"b\"}]}\n"
        /* 2: prompt/completion con cadenas */
        "{\"prompt\": \"2+2?\", \"completion\": \"4\"}\n"
        /* 3: prompt/completion conversacional (listas de mensajes, como TRL) */
        "{\"prompt\": [{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"u\"}], "
        "\"completion\": [{\"role\": \"assistant\", \"content\": \"a\"}]}\n"
        /* 4: Alpaca con input */
        "{\"instruction\": \"Traduce\", \"input\": \"hello\", \"output\": \"hola\"}\n"
        /* 5: Alpaca sin input / input vacio / input null */
        "{\"instruction\": \"Di hola\", \"input\": \"\", \"output\": \"hola\"}\n"
        "{\"instruction\": \"Di chau\", \"output\": \"chau\"}\n"
        "{\"instruction\": \"Di nada\", \"input\": null, \"output\": \"nada\"}\n"
        /* 8: text */
        "{\"text\": \"solo texto\"}\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    assert(buf && count == 9 && skipped == 0);
    const char *want[] = {
        "<|system|>\ns\n<|user|>\nhola\n<|assistant|>\nqu\xc3\xa9 tal\n<|function_call|>\nf()\n<|end|>\n",
        "<|user|>\na\n<|assistant|>\nb\n<|end|>\n",
        "<|user|>\n2+2?\n<|assistant|>\n4\n<|end|>\n",
        "<|system|>\ns\n<|user|>\nu\n<|assistant|>\na\n<|end|>\n",
        "<|user|>\nTraduce\n\nhello\n<|assistant|>\nhola\n<|end|>\n",
        "<|user|>\nDi hola\n<|assistant|>\nhola\n<|end|>\n",
        "<|user|>\nDi chau\n<|assistant|>\nchau\n<|end|>\n",
        "<|user|>\nDi nada\n<|assistant|>\nnada\n<|end|>\n",
        "solo texto",
    };
    for (size_t i = 0; i < 9; i++) {
        char *e = nth_example(buf, len, count, i);
        if (strcmp(e, want[i]) != 0) { printf("FALLA ejemplo %zu:\n%s\n--- esperado:\n%s\n", i, e, want[i]); assert(0); }
        free(e);
    }
    printf("[hf] ShareGPT (from/value y role/content), prompt/completion (cadenas y listas), Alpaca (con/sin input) y text: OK\n");
    free(buf);

    /* registros invalidos: error en strict, saltados en no strict */
    const char *bad[] = {
        "{\"conversations\": []}\n",
        "{\"conversations\": [{\"from\": \"human\"}]}\n",
        "{\"conversations\": [5]}\n",
        "{\"prompt\": \"solo prompt\"}\n",
        "{\"prompt\": 5, \"completion\": \"x\"}\n",
        "{\"instruction\": \"sin salida\"}\n",
        "{\"instruction\": \"x\", \"input\": 5, \"output\": \"y\"}\n",
        "{\"instruction\": 1, \"output\": \"y\"}\n",
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        write_file(path, bad[i]);
        assert(ternary_dataset_load(path, TERNARY_DATASET_JSONL, 1, NULL, 0, &len, &count, &skipped, err, sizeof(err)) == 0);
        buf = load_alloc(path, TERNARY_DATASET_JSONL, 0, &len, &count, &skipped);
        assert(buf && count == 0 && skipped == 1);
        free(buf);
    }
    printf("[hf] 8 casos invalidos: error en strict y 1 saltado en no strict: OK\n");

    /* prioridad cuando un registro trae varias claves: chat > messages > conversations > prompt > instruction > text */
    write_file(path, "{\"text\": \"t\", \"instruction\": \"i\", \"output\": \"o\", \"conversations\": [{\"from\":\"human\",\"value\":\"c\"}]}\n");
    buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    char *e = nth_example(buf, len, count, 0);
    assert(strcmp(e, "<|user|>\nc\n<|end|>\n") == 0);
    free(e); free(buf);
    remove(path);
}

/* ---- razonamiento ----------------------------------------------------------------------------------------------- */
static uint8_t *load_ex(const char *path, int format, int strict, int flags, size_t *len, size_t *count, size_t *skipped) {
    if (!ternary_dataset_load_ex(path, format, strict, flags, NULL, 0, len, count, skipped, err, sizeof(err))) return NULL;
    uint8_t *buf = malloc(*len ? *len : 1);
    assert(ternary_dataset_load_ex(path, format, strict, flags, buf, *len, len, count, skipped, err, sizeof(err)));
    return buf;
}

static void check_reasoning(const char *jsonl, const char *keep, const char *drop) {
    const char *path = tmp_path("test_dataset_razon.jsonl");
    write_file(path, jsonl);
    size_t len, count, skipped;
    for (int mode = 0; mode < 2; mode++) {
        uint8_t *buf = load_ex(path, TERNARY_DATASET_JSONL, 1, mode, &len, &count, &skipped);
        assert(buf && count == 1);
        char *e = nth_example(buf, len, count, 0);
        const char *want = mode ? drop : keep;
        if (strcmp(e, want) != 0) { printf("FALLA (%s):\n%s\n--- esperado:\n%s\n", mode ? "drop" : "keep", e, want); assert(0); }
        free(e); free(buf);
    }
    remove(path);
}

static void test_razonamiento(void) {
    /* etiquetas en el texto, en messages */
    check_reasoning(
        "{\"messages\": [{\"role\": \"user\", \"content\": \"2+2 <think>literal del usuario</think>\"}, "
        "{\"role\": \"assistant\", \"content\": \"<think>\\nsumo\\n</think>\\n\\nSon 4\"}]}\n",
        "<|user|>\n2+2 <think>literal del usuario</think>\n<|assistant|>\n<|think|>\nsumo\n<|/think|>\n\nSon 4\n<|end|>\n",
        "<|user|>\n2+2 <think>literal del usuario</think>\n<|assistant|>\nSon 4\n<|end|>\n");
    /* campo aparte: reasoning_content / reasoning / thinking (el vacio se ignora) */
    const char *fields[] = { "reasoning_content", "reasoning", "thinking" };
    for (int f = 0; f < 3; f++) {
        char js[400];
        snprintf(js, sizeof js, "{\"messages\": [{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"a\", \"%s\": \"pienso\"}]}\n", fields[f]);
        check_reasoning(js, "<|user|>\nq\n<|assistant|>\n<|think|>\npienso\n<|/think|>\n\na\n<|end|>\n", "<|user|>\nq\n<|assistant|>\na\n<|end|>\n");
    }
    check_reasoning("{\"messages\": [{\"role\": \"assistant\", \"content\": \"a\", \"reasoning\": \"\"}]}\n",
                    "<|assistant|>\na\n<|end|>\n", "<|assistant|>\na\n<|end|>\n");
    /* <thinking>, literales <|think|> y varios bloques */
    check_reasoning("{\"messages\": [{\"role\": \"assistant\", \"content\": \"<thinking>x</thinking>A<|think|>y<|/think|>B\"}]}\n",
                    "<|assistant|>\n<|think|>x<|/think|>A<|think|>y<|/think|>B\n<|end|>\n", "<|assistant|>\nAB\n<|end|>\n");
    /* chat (model), ShareGPT (gpt), completion, Alpaca */
    check_reasoning("{\"chat\": [{\"user\": \"q\"}, {\"model\": \"<think>r</think>a\"}]}\n",
                    "<|user|>\nq\n<|assistant|>\n<|think|>r<|/think|>a\n<|end|>\n", "<|user|>\nq\n<|assistant|>\na\n<|end|>\n");
    check_reasoning("{\"conversations\": [{\"from\": \"human\", \"value\": \"q\"}, {\"from\": \"gpt\", \"value\": \"<think>r</think>a\"}]}\n",
                    "<|user|>\nq\n<|assistant|>\n<|think|>r<|/think|>a\n<|end|>\n", "<|user|>\nq\n<|assistant|>\na\n<|end|>\n");
    check_reasoning("{\"prompt\": \"q\", \"completion\": \"a\", \"reasoning\": \"r\"}\n",
                    "<|user|>\nq\n<|assistant|>\n<|think|>\nr\n<|/think|>\n\na\n<|end|>\n", "<|user|>\nq\n<|assistant|>\na\n<|end|>\n");
    check_reasoning("{\"instruction\": \"q\", \"output\": \"<think>r</think> a\", \"thinking\": \"t\"}\n",
                    "<|user|>\nq\n<|assistant|>\n<|think|>\nt\n<|/think|>\n\n<|think|>r<|/think|> a\n<|end|>\n", "<|user|>\nq\n<|assistant|>\na\n<|end|>\n");
    /* 'text' no se toca */
    check_reasoning("{\"text\": \"un <think>texto</think> plano\"}\n", "un <think>texto</think> plano", "un <think>texto</think> plano");
    printf("[razonamiento] etiquetas y campos (messages, chat, ShareGPT, prompt/completion, Alpaca): keep y drop: OK\n");

    /* mal balanceado: error en strict, saltado en no strict, igual con keep y con drop */
    const char *bad[] = {
        "{\"messages\": [{\"role\": \"assistant\", \"content\": \"<think>sin cerrar\"}]}\n",
        "{\"messages\": [{\"role\": \"assistant\", \"content\": \"cierre solo</think>\"}]}\n",
        "{\"messages\": [{\"role\": \"assistant\", \"content\": \"<think>a<think>b</think></think>\"}]}\n",
        "{\"chat\": [{\"model\": \"<|/think|>x\"}]}\n",
    };
    const char *path = tmp_path("test_dataset_razon_mal.jsonl");
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        write_file(path, bad[i]);
        for (int mode = 0; mode < 2; mode++) {
            size_t len, count, skipped;
            assert(ternary_dataset_load_ex(path, TERNARY_DATASET_JSONL, 1, mode, NULL, 0, &len, &count, &skipped, err, sizeof(err)) == 0);
            assert(strstr(err, "razonamiento") != NULL);
            uint8_t *buf = load_ex(path, TERNARY_DATASET_JSONL, 0, mode, &len, &count, &skipped);
            assert(buf && count == 0 && skipped == 1);
            free(buf);
        }
    }
    remove(path);
    printf("[razonamiento] etiquetas mal balanceadas: error en strict y saltado en no strict: OK\n");
    /* ternary_dataset_load (sin flags) == KEEP */
    write_file(path, "{\"messages\": [{\"role\": \"assistant\", \"content\": \"<think>r</think>a\"}]}\n");
    size_t len, count, skipped;
    uint8_t *buf = load_alloc(path, TERNARY_DATASET_JSONL, 1, &len, &count, &skipped);
    char *e = nth_example(buf, len, count, 0);
    assert(strcmp(e, "<|assistant|>\n<|think|>r<|/think|>a\n<|end|>\n") == 0);
    free(e); free(buf); remove(path);
}

int main(void) {
    test_texto_lineas_vs_bloque();
    test_jsonl_renderiza_turnos();
    test_json_arreglo_y_utf8();
    test_strict_vs_no_strict();
    test_json_array_no_strict_saltea_elemento();
    test_chat_propio();
    test_chat_roles();
    test_messages_roles_y_text();
    test_formatos_hf();
    test_razonamiento();
    printf("\ntest_dataset: TODO OK\n");
    return 0;
}
