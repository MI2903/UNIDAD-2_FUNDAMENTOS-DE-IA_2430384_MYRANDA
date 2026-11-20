"""Genera capturas de cada relajación de Bellman-Ford.

Sin argumentos genera 8 -> 24 y 24 -> 8 en el grafo positivo. Se usa el
grafo positivo porque, en el grafo negativo dirigido, 8 -> 24 no existe.
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path

from matplotlib.patches import FancyArrowPatch

from generar_capturas_dijkstra import AZUL, DESCARTADA, MEJORA, DibujadorBusqueda
from grafo_visual import POSICIONES, arcos, bellman_ford, construir_grafo, reconstruir


RAIZ = Path(__file__).resolve().parent
COLOR_ARBOL = "#168369"


@dataclass
class EventoBellman:
    tipo: str
    pasada: int
    numero_arco: int
    a: int | None
    b: int | None
    clave: str | None
    peso: float | None
    anterior: float | None
    propuesta: float | None
    mejora: bool | None
    distancias: dict[int, float]
    antecesores: dict[int, tuple[int, str]]
    cambios_pasada: int = 0


def rastrear_bellman(grafo, origen):
    """Repite el orden de ``bellman_ford`` y conserva cada decisión."""
    distancias = {n: math.inf for n in grafo.nodes}
    distancias[origen] = 0
    antecesores = {}
    lista_arcos = list(arcos(grafo))
    eventos = [EventoBellman("inicio", 0, 0, None, None, None, None,
                             None, None, None, distancias.copy(), {})]

    for pasada in range(1, len(grafo.nodes)):
        cambios = 0
        for numero, (a, b, clave, peso) in enumerate(lista_arcos, 1):
            anterior = distancias[b]
            propuesta = (distancias[a] + peso if not math.isinf(distancias[a])
                         else math.inf)
            mejora = propuesta < anterior
            if mejora:
                distancias[b] = propuesta
                antecesores[b] = (a, clave)
                cambios += 1
            eventos.append(EventoBellman(
                "relajacion", pasada, numero, a, b, clave, peso,
                anterior, propuesta, mejora, distancias.copy(),
                antecesores.copy(), cambios,
            ))
        eventos.append(EventoBellman(
            "fin_pasada", pasada, len(lista_arcos), None, None, None, None,
            None, None, None, distancias.copy(), antecesores.copy(), cambios,
        ))
        if not cambios:
            break

    # La última comprobación del algoritmo debe verificar todos los arcos.
    for a, b, _, peso in lista_arcos:
        if not math.isinf(distancias[a]) and distancias[a] + peso < distancias[b]:
            raise ValueError("Hay un ciclo negativo alcanzable desde el origen.")
    eventos.append(EventoBellman(
        "verificacion", pasada, len(lista_arcos), None, None, None, None,
        None, None, None, distancias.copy(), antecesores.copy(), 0,
    ))
    return eventos, lista_arcos


def numero(valor):
    return "∞" if valor is None or math.isinf(valor) else f"{valor:g}"


class DibujadorBellman(DibujadorBusqueda):
    def __init__(self, grafo, origen, destino):
        super().__init__(grafo, origen, destino)
        self.leyenda.set_text(
            "Verde: mejor antecesor conocido  ·  ámbar: distancia conocida  ·  "
            "naranja: mejora  ·  morado: sin mejora"
        )
        self.tabla_titulo.set_text("Distancia provisional y antecesor tras cada relajación")

    def flecha_actual(self, a, b, clave, color, punteada=False):
        curva = self.curvas.get(clave, 0)
        origen_fisico, _ = self.extremos[clave]
        if a != origen_fisico:
            curva = -curva
        for borde in (True, False):
            parche = FancyArrowPatch(
                POSICIONES[a], POSICIONES[b],
                arrowstyle="-|>", mutation_scale=20,
                connectionstyle=f"arc3,rad={curva}",
                linewidth=8.0 if borde else 5.0,
                color="white" if borde else color,
                linestyle="--" if punteada and not borde else "-",
                shrinkA=22, shrinkB=22, capstyle="round", zorder=6 if borde else 7,
            )
            self.ax.add_patch(parche)
            self.dinamicos.append(parche)

    def mostrar_bellman(self, evento, indice, total, archivo, total_arcos,
                        ruta_definitiva=None):
        self.limpiar_dinamicos()
        for peso in self.pesos.values():
            peso.set_color(AZUL)

        # El árbol de antecesores muestra las rutas provisionales que el
        # algoritmo ha encontrado hasta este instante.
        for b, (a, clave) in evento.antecesores.items():
            curva = self.curvas.get(clave, 0)
            if a != self.extremos[clave][0]:
                curva = -curva
            self.dibujar_arista(a, b, COLOR_ARBOL, 3.4, 3,
                                curva)
            self.pesos[clave].set_color(COLOR_ARBOL)

        if ruta_definitiva is not None:
            self.titulo.set_text(
                f"Bellman-Ford {self.origen} → {self.destino} · ruta definitiva"
            )
            self.detalle.set_text(
                f"Costo mínimo {numero(ruta_definitiva.distancia)} · "
                f"{evento.pasada} pasadas · sin ciclo negativo"
            )
            self.camino_texto.set_text(
                "Ruta óptima: " + " → ".join(map(str, ruta_definitiva.nodos))
            )
            for a, b, clave in zip(
                ruta_definitiva.nodos, ruta_definitiva.nodos[1:],
                ruta_definitiva.aristas,
            ):
                self.flecha_actual(a, b, clave, MEJORA)
                self.pesos[clave].set_color(MEJORA)
        else:
            self.titulo.set_text(
                f"Bellman-Ford {self.origen} → {self.destino} · "
                f"paso {indice:04d}/{total:04d}"
            )
            if evento.tipo == "inicio":
                self.detalle.set_text(
                    "Inicialización: d[origen] = 0; los demás nodos = ∞"
                )
                self.camino_texto.set_text(f"Comienza en el nodo {self.origen}")
            elif evento.tipo == "relajacion":
                decision = ("mejora" if evento.mejora else
                            "sin cambio: origen aún en ∞" if math.isinf(evento.propuesta)
                            else "sin mejora")
                self.detalle.set_text(
                    f"Pasada {evento.pasada} · arco {evento.numero_arco}/{total_arcos} "
                    f"· {evento.a} → {evento.b} (peso {numero(evento.peso)}) "
                    f"· propuesta {numero(evento.propuesta)} "
                    f"frente a {numero(evento.anterior)} · {decision}"
                )
                camino = reconstruir(
                    self.origen, evento.b, evento.antecesores,
                    evento.distancias[evento.b],
                )
                self.camino_texto.set_text(
                    "Mejor camino conocido a este destino: " +
                    (" → ".join(map(str, camino.nodos)) if camino else "aún no existe")
                )
                color = MEJORA if evento.mejora else DESCARTADA
                self.flecha_actual(evento.a, evento.b, evento.clave, color,
                                   punteada=not evento.mejora)
                self.pesos[evento.clave].set_color(color)
            elif evento.tipo == "fin_pasada":
                self.detalle.set_text(
                    f"Fin de la pasada {evento.pasada}: "
                    f"{evento.cambios_pasada} distancias mejoradas"
                )
                self.camino_texto.set_text(
                    "Se repetirá la revisión de aristas" if evento.cambios_pasada else
                    "No hubo mejoras: la búsqueda terminó"
                )
            else:
                self.detalle.set_text(
                    f"Verificación final: {total_arcos} arcos revisados; "
                    "no hay ciclo negativo alcanzable"
                )
                self.camino_texto.set_text("Las distancias y antecesores ya son definitivos")

        colores = []
        for nodo in self.nodos:
            color = "#e4a846" if not math.isinf(evento.distancias[nodo]) else "#356ca1"
            if nodo == self.origen:
                color = "#168369"
            if evento.tipo == "relajacion" and nodo == evento.b:
                color = MEJORA if evento.mejora else DESCARTADA
            if ruta_definitiva is not None and nodo in ruta_definitiva.nodos:
                color = "#e4a846"
            if ruta_definitiva is not None and nodo == self.destino:
                color = "#b6533d"
            if nodo == self.origen:
                color = "#168369"
            colores.append(color)
        self.marca_nodos.set_facecolors(colores)

        for nodo, etiqueta in self.tabla.items():
            ant = evento.antecesores[nodo][0] if nodo in evento.antecesores else "—"
            etiqueta.set_text(
                f"{nodo:>2}   d={numero(evento.distancias[nodo]):<4}  ant={ant}"
            )
            etiqueta.set_color(COLOR_ARBOL if nodo in evento.antecesores else AZUL)

        self.figura.savefig(archivo, dpi=120, facecolor="white")


def nombre_evento(indice, evento):
    if evento.tipo == "inicio":
        return f"paso_{indice:04d}_inicio.png"
    if evento.tipo == "relajacion":
        decision = "mejora" if evento.mejora else "sin_mejora"
        return (f"paso_{indice:04d}_pasada_{evento.pasada:02d}_"
                f"arco_{evento.numero_arco:03d}_{evento.a}_a_{evento.b}_{decision}.png")
    if evento.tipo == "fin_pasada":
        return f"paso_{indice:04d}_fin_pasada_{evento.pasada:02d}.png"
    return f"paso_{indice:04d}_verificacion_final.png"


def crear_indice(carpeta, archivos, origen, destino):
    nombres = [p.name for p in archivos]
    opciones = "\n".join(
        f'<option value="{i}">{i:04d} · {html.escape(nombre)}</option>'
        for i, nombre in enumerate(nombres)
    )
    contenido = f"""<!doctype html>
<html lang="es"><meta charset="utf-8"><title>Bellman-Ford {origen} → {destino}</title>
<style>body{{margin:0;background:#eaf0f6;color:#173456;font:16px sans-serif}}
header{{position:sticky;top:0;padding:12px;background:white;display:flex;gap:12px;
align-items:center;box-shadow:0 2px 9px #0002}}button,select{{font:inherit;padding:7px}}
img{{display:block;width:min(100%,1800px);margin:12px auto;box-shadow:0 3px 16px #0002}}
</style><header><strong>Bellman-Ford {origen} → {destino}</strong>
<button id="prev">← Anterior</button><select id="paso">{opciones}</select>
<button id="next">Siguiente →</button><span>Usa ← y → en el teclado</span></header>
<img id="captura" alt="Captura de Bellman-Ford">
<script>const selector=document.querySelector('#paso'),img=document.querySelector('#captura');
function mostrar(n){{selector.selectedIndex=Math.max(0,Math.min(n,selector.options.length-1));
img.src=selector.options[selector.selectedIndex].textContent.split(' · ').slice(1).join(' · ');}}
selector.onchange=()=>mostrar(selector.selectedIndex);
document.querySelector('#prev').onclick=()=>mostrar(selector.selectedIndex-1);
document.querySelector('#next').onclick=()=>mostrar(selector.selectedIndex+1);
document.onkeydown=e=>{{if(e.key==='ArrowLeft')mostrar(selector.selectedIndex-1);
if(e.key==='ArrowRight')mostrar(selector.selectedIndex+1)}};mostrar(0);</script></html>"""
    (carpeta / "indice.html").write_text(contenido, encoding="utf-8")


def generar_capturas(origen, destino, raiz=RAIZ):
    grafo = construir_grafo(False)
    if origen not in grafo or destino not in grafo:
        raise ValueError("El origen y el destino deben existir en el grafo.")
    eventos, lista_arcos = rastrear_bellman(grafo, origen)
    ruta = reconstruir(origen, destino, eventos[-1].antecesores,
                        eventos[-1].distancias[destino])
    referencia = bellman_ford(grafo, origen, destino)
    if ruta != referencia:
        raise AssertionError("La traza no coincide con bellman_ford.")
    if ruta is None:
        raise ValueError(f"No hay ruta entre {origen} y {destino}.")

    raiz = Path(raiz).resolve()
    carpeta = raiz / f"capturas_nodo_inicial_{origen}_bellman"
    if not carpeta.is_relative_to(raiz):
        raise ValueError("La carpeta debe estar dentro del proyecto.")
    carpeta.mkdir(parents=True, exist_ok=True)
    for patron in ("paso_*.png", "ruta_definitiva_*.png"):
        for anterior in carpeta.glob(patron):
            if anterior.is_file() and anterior.resolve().parent == carpeta:
                anterior.unlink()

    dibujador = DibujadorBellman(grafo, origen, destino)
    archivos = []
    resumen = [f"Bellman-Ford: {origen} → {destino}",
               "Grafo positivo no dirigido: cada arista se evalúa en ambos sentidos.",
               f"Arcos evaluados por pasada: {len(lista_arcos)}",
               f"Pasadas: {eventos[-1].pasada}",
               f"Relajaciones examinadas: {sum(e.tipo == 'relajacion' for e in eventos)}",
               "Ruta definitiva: " + " → ".join(map(str, ruta.nodos)),
               f"Costo mínimo: {numero(ruta.distancia)}", ""]
    total = len(eventos) - 1
    for indice, evento in enumerate(eventos):
        archivo = carpeta / nombre_evento(indice, evento)
        dibujador.mostrar_bellman(evento, indice, total, archivo, len(lista_arcos))
        archivos.append(archivo)
        if evento.tipo == "relajacion":
            decision = "mejora" if evento.mejora else "sin mejora"
            resumen.append(
                f"{archivo.name}: d[{evento.b}]={numero(evento.anterior)}, "
                f"d[{evento.a}]+{numero(evento.peso)}={numero(evento.propuesta)} "
                f"→ {decision}; nuevo d[{evento.b}]={numero(evento.distancias[evento.b])}"
            )
        elif evento.tipo == "fin_pasada":
            resumen.append(f"{archivo.name}: {evento.cambios_pasada} mejoras")
        else:
            resumen.append(f"{archivo.name}: {evento.tipo}")
        if indice and indice % 126 == 0:
            print(f"  {origen} -> {destino}: {indice}/{total} pasos", flush=True)

    definitiva = carpeta / f"ruta_definitiva_{origen}_a_{destino}.png"
    dibujador.mostrar_bellman(eventos[-1], len(eventos), len(eventos),
                              definitiva, len(lista_arcos), ruta)
    archivos.append(definitiva)
    resumen.append(f"{definitiva.name}: costo {numero(ruta.distancia)}")
    (carpeta / "resumen.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    crear_indice(carpeta, archivos, origen, destino)
    dibujador.figura.clear()
    return archivos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origen", nargs="?", type=int)
    parser.add_argument("destino", nargs="?", type=int)
    args = parser.parse_args()
    if (args.origen is None) != (args.destino is None):
        parser.error("Escribe ambos nodos o ninguno.")
    pares = [(8, 24), (24, 8)] if args.origen is None else [(args.origen, args.destino)]
    for origen, destino in pares:
        archivos = generar_capturas(origen, destino)
        print(f"{origen} -> {destino}: {len(archivos)} imagenes en {archivos[0].parent}")


if __name__ == "__main__":
    main()
