"""Igualdad bit a bit del entrenamiento entre variantes de núcleo y con los valores de referencia.

Entrena de forma determinista seis modelos pequeños (ReLU² y SiLU, ventana con sinks, dimensiones que no son múltiplo de 4 ni de 16,
smear) con 1 y 2 hilos, y compara tres huellas (historia, logits y archivos guardados) entre:
  - el camino por defecto (AVX-512 + VNNI si la CPU los tiene),
  - TERNARYAI_NO_VNNI=1 (AVX-512 sin el forward entero VNNI),
  - TERNARYAI_NO_AVX512=1 (núcleo escalar).
Las tres variantes deben dar las mismas huellas, y la historia y los logits deben coincidir con los valores de referencia de abajo
(calculados con la versión 1.0.0a1 original; el backward de pesos usa multiplicación y suma fusionadas y puede cambiar el estado
guardado, pero en estos entrenamientos cortos no cambia la historia ni los logits). Si una mejora de velocidad cambia un bit, esta prueba falla.

Uso:  python3 tests/test_bits.py        (con ternaryai importable: pip install . o PYTHONPATH=python)
"""
import glob
import hashlib
import os
import subprocess
import sys
import tempfile

# (hilos, modelo) -> (huella de la historia, huella de los logits)
REFERENCIA = {
    "lm_relu2": ("19b5641a3f46dbfc", "1f974ca8ba665c4a"),
    "lm_win": ("fe4acd78da7d2ab1", "b9ba84e4051e08b4"),
    "lm_odd": ("f5491a969517e669", "3ec0c465f35f4711"),
    "lm_win2": ("0e2fc8facc5c11ca", "7ab6dcb3383048df"),
    "lm_odd2": ("c7ce9d8869034783", "1b136532f3afd620"),
    "lm_silu": ("01404174878786a3", "5cb181eaaa663280"),
}


def _h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def trabajador():
    import numpy as np
    import ternaryai as tn

    def bloque(**kw):
        return tn.transformer_block(**kw)

    modelos = {
        "lm_relu2": (32, 256, lambda: tn.Model(32, [tn.Embedding(256, 64), *bloque(heads=4, hidden=96, kv_heads=2, gate_mode=2) * 2, tn.RMSNorm(activation="uint8"), tn.LMHead(256)], seed=3)),
        "lm_win": (32, 256, lambda: tn.Model(32, [tn.Embedding(256, 64), *bloque(heads=4, hidden=96, kv_heads=2, window=8, sinks=2, gate_mode=2) * 2, tn.RMSNorm(activation="uint8"), tn.LMHead(256)], seed=9)),
        "lm_odd": (40, 300, lambda: tn.Model(40, [tn.Embedding(300, 72), *bloque(heads=3, hidden=100, kv_heads=1, smear=True, gate_mode=2) * 2, tn.RMSNorm(activation="uint8"), tn.LMHead(300)], seed=11)),
        "lm_win2": (96, 200, lambda: tn.Model(96, [tn.Embedding(200, 64), *bloque(heads=4, hidden=96, kv_heads=2, window=24, sinks=3, gate_mode=2) * 2, tn.RMSNorm(activation="uint8"), tn.LMHead(200)], seed=17)),
        "lm_odd2": (24, 130, lambda: tn.Model(24, [tn.Embedding(130, 70), *bloque(heads=5, hidden=99, kv_heads=1, gate_mode=2) * 2, tn.RMSNorm(activation="uint8"), tn.LMHead(130)], seed=13)),
        "lm_silu": (32, 256, lambda: tn.Model(32, [tn.Embedding(256, 64), *bloque(heads=4, hidden=96, kv_heads=2) * 2, tn.RMSNorm(activation="uint8"), tn.LMHead(256)], seed=5)),
    }
    for hilos in (1, 2):
        tn.set_threads(hilos)
        for nombre, (seq, vocab, crear) in modelos.items():
            rng = np.random.default_rng(7)
            x = rng.integers(0, vocab, (24, seq), dtype=np.int32)
            y = np.roll(x, -1, 1)
            y[:, :3] = -1
            m = crear()
            hist = m.fit(x, y, epochs=4)
            logits = np.ascontiguousarray(m.predict(x[:4]))
            with tempfile.TemporaryDirectory() as d:
                m.save(d)
                archivos = b"".join(open(f, "rb").read() for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f))
            print(f"{hilos} {nombre} {_h(repr(hist).encode())} {_h(logits.tobytes())} {_h(archivos)}")


def correr(env_extra):
    env = dict(os.environ)
    for k in ("TERNARYAI_NO_AVX512", "TERNARYAI_NO_VNNI"):
        env.pop(k, None)
    env.update(env_extra)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--trabajador"], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("FALLA: el trabajador terminó con error")
    return [tuple(l.split()) for l in r.stdout.strip().splitlines()]


def main():
    variantes = {"por defecto": {}, "sin VNNI": {"TERNARYAI_NO_VNNI": "1"}, "escalar": {"TERNARYAI_NO_AVX512": "1"}}
    resultados = {nombre: correr(env) for nombre, env in variantes.items()}
    base = resultados["por defecto"]
    ok = True
    for nombre, res in resultados.items():
        if res != base:
            ok = False
            print(f"FALLA: la variante '{nombre}' difiere de la de por defecto")
            for a, b in zip(base, res):
                if a != b:
                    print("  por defecto:", a, "\n  ", nombre + ":", b)
    for hilos, modelo, hist, logits, _archivos in base:
        esperado = REFERENCIA[modelo]
        if (hist, logits) != esperado:
            ok = False
            print(f"FALLA: {modelo} (hilos={hilos}) no coincide con la referencia 1.0.0a1: {(hist, logits)} != {esperado}")
    print(("OK  " if ok else "MAL ") + f"test_bits ({len(base)} entrenamientos x {len(variantes)} variantes)")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--trabajador" in sys.argv:
        trabajador()
    else:
        sys.exit(main())
