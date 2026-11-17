#!/bin/sh
# Prueba de funcionalidad (despues de: make -j8 all). Uso: sh probar.sh [largo]
# Usa TERNARYAI_THREADS=6 (el resultado es identico bit a bit con cualquier numero de hilos).
cd "$(dirname "$0")"
[ -x tests/test_core ] || { echo "Faltan los ejecutables: compile primero con make (ver LEEME)"; exit 1; }
export TMPDIR="${TMPDIR:-$HOME/tmp}"; mkdir -p "$TMPDIR"
export TERNARYAI_THREADS="${TERNARYAI_THREADS:-6}"
ok=0; mal=0
echo "CPU con NEON: $(grep -o -m1 -w neon /proc/cpuinfo | head -1)  hilos=$TERNARYAI_THREADS"
LISTA=$(ls tests | grep '^test_' | grep -v '\.' | grep -v -e test_avx512 -e test_infer$ -e test_simple)
[ "$1" = "largo" ] || LISTA=$(echo "$LISTA" | grep -v -e gqa_learn -e gqa_window -e test_blocks -e test_gqa_scaling)
for t in $LISTA; do
  if out=$(./tests/$t 2>&1); then ok=$((ok+1)); printf "OK     %-26s %s\n" "$t" "$(echo "$out" | tail -1 | cut -c1-50)"
  else mal=$((mal+1)); printf "FALLA  %-26s %s\n" "$t" "$(echo "$out" | tail -1 | cut -c1-50)"; fi
done
for m in relu2 wide; do
  if out=$(./tests/test_infer tests/modelos_prueba/$m 2>&1); then ok=$((ok+1)); echo "OK     test_infer $m: $(echo "$out" | tail -1)"
  else mal=$((mal+1)); echo "FALLA  test_infer $m: $(echo "$out" | tail -2 | tr '\n' ' ')"; fi
done
echo "Resultado: $ok correctos, $mal con fallas (test_simple se omite: compara con rand() de glibc)"
