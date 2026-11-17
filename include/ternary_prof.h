#ifndef TERNARY_PROF_H
#define TERNARY_PROF_H
/* Perfilador por ambitos (arbol de llamadas) con cronometro de pared, apagado por defecto.
 *
 * Uso en el codigo: TPROF_BEGIN("nombre"); ...; TPROF_END();  (o TPROF_BEGIN_I("nombre", indice) para distinguir
 * instancias, p. ej. la capa 3 de un bloque). Los ambitos se anidan: cada nodo del arbol guarda llamadas, tiempo
 * TOTAL (incluye hijos) y tiempo PROPIO (total menos hijos), asi la suma de los tiempos propios es el tiempo medido.
 *
 * Apagado cuesta una comparacion de un entero por ambito. No toca ningun dato del calculo (mismos bits).
 * Solo cuenta el hilo que lo activo (los hilos del pool se ignoran; el tiempo de pared del llamador ya los incluye).
 * Activar: ternaryai_prof_enable(1) (API plana) / Model.prof_enable() en Python. */
#include <stddef.h>
#include <stdint.h>

extern int g_tprof_on;
void ternary_prof_enter(const char *name, int idx);
void ternary_prof_leave(void);
void ternary_prof_enable(int on);               /* al activar, el hilo que llama pasa a ser el unico que cuenta */
void ternary_prof_reset(void);                  /* pone en cero llamadas y tiempos (conserva el arbol) */
/* Arbol en JSON: {"overhead_ns_per_scope":x,"nodes":[{"id":..,"parent":..,"name":..,"idx":..,"calls":..,"total_ns":..,"self_ns":..}, ...]}
 * Devuelve el largo total necesario (sin el NUL); escribe a lo sumo cap-1 bytes y termina en NUL. */
size_t ternary_prof_dump_json(char *buf, size_t cap);

#define TPROF_BEGIN(name)      do { if (g_tprof_on) ternary_prof_enter((name), -1); } while (0)
#define TPROF_BEGIN_I(name, i) do { if (g_tprof_on) ternary_prof_enter((name), (int)(i)); } while (0)
#define TPROF_END()            do { if (g_tprof_on) ternary_prof_leave(); } while (0)

#endif
