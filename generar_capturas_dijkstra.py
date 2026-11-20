"""Captura toda la búsqueda de Dijkstra y, al final, su ruta definitiva.

Sin argumentos genera 24 -> 8 y 8 -> 24. También admite
``python generar_capturas_dijkstra.py ORIGEN DESTINO``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch

from grafo_visual import (
    POSICIONES, construir_grafo, curvaturas_del_grafo, dijkstra, reconstruir,
)


RAIZ = Path(__file__).resolve().parent
AZUL = "#173456"
ARISTA = "#91a8be"
MEJORA = "#e65a37"
DESCARTADA = "#96567b"


def se_cruzan(caja_a, caja_b, margen=0):
    return (caja_a[0] < caja_b[2] + margen and
            caja_a[2] > caja_b[0] - margen and
            caja_a[1] < caja_b[3] + margen and
            caja_a[3] > caja_b[1] - margen)


def segmento_cruza_caja(a, b, caja):
    """Comprueba si un tramo cruza un rectángulo mediante recorte paramétrico."""
    x0, y0 = a
    dx, dy = b[0] - x0, b[1] - y0
    xmin, ymin, xmax, ymax = caja
    entrada, salida = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) < 1e-9:
            if q < 0:
                return False
        else:
            fraccion = q / p
            if p < 0:
                entrada = max(entrada, fraccion)
            else:
                salida = min(salida, fraccion)
            if entrada > salida:
                return False
    return True


class DibujadorBusqueda:
    """Mantiene fijos los 63 pesos y actualiza solo el estado de cada decisión."""

    def __init__(self, grafo, origen, destino):
        self.grafo = grafo
        self.origen = origen
        self.destino = destino
        self.curvas = curvaturas_del_grafo(grafo)
        self.aristas = list(grafo.edges(keys=True, data=True))
        self.extremos = {clave: (a, b) for a, b, clave, _ in self.aristas}
        self.figura = Figure(figsize=(20, 11.25), dpi=120, facecolor="white")
        self.canvas = FigureCanvasAgg(self.figura)
        self.ax = self.figura.add_axes((0.035, 0.265, 0.93, 0.615))
        self.ax.set_xlim(-0.65, 14.95)
        self.ax.set_ylim(-0.45, 7.20)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")
        self.parches = {}
        for a, b, clave, _ in self.aristas:
            self.parches[clave] = self.dibujar_arista(
                a, b, ARISTA, 2.0, 1, self.curvas.get(clave, 0)
            )

        self.nodos = list(grafo.nodes)
        xy = [POSICIONES[n] for n in self.nodos]
        self.marca_nodos = self.ax.scatter(
            [p[0] for p in xy], [p[1] for p in xy], s=690,
            c=["#356ca1"] * len(xy), edgecolors="white",
            linewidths=1.9, zorder=8,
        )
        for nodo, (x, y) in zip(self.nodos, xy):
            self.ax.text(x, y, str(nodo), ha="center", va="center",
                         color="white", fontsize=10.5, fontweight="bold", zorder=9)

        self.canvas.draw()
        posiciones = self.ubicar_todos_los_pesos()
        self.pesos = {}
        for _, _, clave, datos in self.aristas:
            x, y = posiciones[clave]
            self.pesos[clave] = self.ax.text(
                x, y, str(datos["weight"]), ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=AZUL, zorder=12,
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white",
                      "edgecolor": "#dae4ed", "linewidth": 0.5, "alpha": 0.96},
            )

        self.titulo = self.figura.text(0.045, 0.953, "", fontsize=22,
                                      fontweight="bold", color=AZUL)
        self.detalle = self.figura.text(0.045, 0.920, "", fontsize=13.5,
                                       color="#46617c")
        self.camino_texto = self.figura.text(0.045, 0.225, "", fontsize=13.5,
                                            fontweight="bold", color=AZUL)
        self.leyenda = self.figura.text(
            0.045, 0.195,
            "Verde: confirmado  ·  ámbar: provisional  ·  naranja: mejora  ·  morado: sin mejora",
            fontsize=10.5, color="#48627b",
        )
        self.tabla_titulo = self.figura.text(
            0.045, 0.155, "Distancia provisional y antecesor de cada nodo",
            fontsize=12, fontweight="bold", color=AZUL,
        )
        self.tabla = {}
        for indice, nodo in enumerate(sorted(grafo.nodes)):
            columna, fila = indice % 7, indice // 7
            self.tabla[nodo] = self.figura.text(
                0.045 + columna * 0.133, 0.126 - fila * 0.026,
                "", fontsize=10.2, color=AZUL,
            )
        self.dinamicos = []

    def dibujar_arista(self, a, b, color, ancho, capa, curva=0, estilo="-"):
        for borde in (True, False):
            parche = FancyArrowPatch(
                POSICIONES[a], POSICIONES[b], arrowstyle="-",
                connectionstyle=f"arc3,rad={curva}",
                linewidth=ancho + 2.9 if borde else ancho,
                color="white" if borde else color,
                linestyle=estilo if not borde else "-",
                shrinkA=20, shrinkB=20, capstyle="round",
                zorder=capa if borde else capa + 0.1,
            )
            self.ax.add_patch(parche)
            if borde:
                self.dinamicos.append(parche) if hasattr(self, "dinamicos") else None
            else:
                self.dinamicos.append(parche) if hasattr(self, "dinamicos") else None
        return parche

    def ubicar_todos_los_pesos(self):
        """Busca posiciones de etiquetas sin tapar nodos ni otros pesos."""
        renderer = self.canvas.get_renderer()
        fuente = FontProperties(family="DejaVu Sans", weight="bold", size=10.5)
        nodos_px = [self.ax.transData.transform(POSICIONES[n]) for n in self.nodos]
        radio = math.sqrt(690 / math.pi) * self.figura.dpi / 72 + 5
        limites = self.ax.bbox
        curvas = {}
        tramos = {}
        for _, _, clave, _ in self.aristas:
            vertices = self.parches[clave].get_path().transformed(
                self.parches[clave].get_transform()
            ).vertices
            p0, p1, p2 = vertices[:3]
            curvas[clave] = (p0, p1, p2)
            puntos = []
            for paso in range(17):
                t = paso / 16
                puntos.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
                               (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]))
            tramos[clave] = [(a, b, (min(a[0], b[0]), min(a[1], b[1]),
                                      max(a[0], b[0]), max(a[1], b[1])))
                             for a, b in zip(puntos, puntos[1:])]

        colocadas = []
        posiciones = {}
        for _, _, clave, datos in sorted(
            self.aristas,
            key=lambda e: math.dist(POSICIONES[e[0]], POSICIONES[e[1]]),
        ):
            texto = str(datos["weight"])
            ancho, alto, _ = renderer.get_text_width_height_descent(texto, fuente, False)
            ancho, alto = ancho + 14, alto + 12
            p0, p1, p2 = curvas[clave]
            mejor = None
            for t in (0.5, 0.42, 0.58, 0.34, 0.66, 0.25, 0.75, 0.16, 0.84):
                x0 = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
                y0 = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
                tx = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
                ty = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
                longitud = math.hypot(tx, ty) or 1
                nx, ny = -ty / longitud, tx / longitud
                for separacion in (0, 14, -14, 27, -27, 41, -41, 56, -56,
                                   74, -74, 94, -94, 118, -118):
                    x, y = x0 + nx * separacion, y0 + ny * separacion
                    caja = (x - ancho / 2, y - alto / 2,
                            x + ancho / 2, y + alto / 2)
                    if (caja[0] < limites.x0 + 4 or caja[2] > limites.x1 - 4 or
                            caja[1] < limites.y0 + 4 or caja[3] > limites.y1 - 4):
                        continue
                    if any((px - min(max(px, caja[0]), caja[2])) ** 2 +
                           (py - min(max(py, caja[1]), caja[3])) ** 2 < radio ** 2
                           for px, py in nodos_px):
                        continue
                    if any(se_cruzan(caja, otra, 5) for otra in colocadas):
                        continue
                    cruces = sum(
                        1 for otra, segmentos in tramos.items() if otra != clave
                        for a, b, envolvente in segmentos
                        if se_cruzan(caja, envolvente, 2) and
                        segmento_cruza_caja(a, b, caja)
                    )
                    puntuacion = cruces * 130 + abs(separacion) * 1.5 + abs(t - 0.5) * 75
                    if mejor is None or puntuacion < mejor[0]:
                        mejor = (puntuacion, x, y, caja)
                    if puntuacion == 0:
                        break
                if mejor is not None and mejor[0] == 0:
                    break
            if mejor is None:
                raise RuntimeError(f"No hay espacio para el peso de la arista {clave}")
            _, px, py, caja = mejor
            colocadas.append(caja)
            posiciones[clave] = self.ax.transData.inverted().transform((px, py))
        return posiciones

    def limpiar_dinamicos(self):
        for artista in self.dinamicos:
            artista.remove()
        self.dinamicos.clear()

    def camino_hasta(self, nodo, evento):
        return reconstruir(self.origen, nodo, evento.antecesores,
                           evento.distancias[nodo])

    def mostrar(self, evento, indice, total, archivo, ruta_definitiva=None):
        self.limpiar_dinamicos()
        for peso in self.pesos.values():
            peso.set_color(AZUL)

        if ruta_definitiva is not None:
            nodos_camino = ruta_definitiva.nodos
            aristas_camino = ruta_definitiva.aristas
            self.titulo.set_text(f"Dijkstra {self.origen} → {self.destino} · ruta definitiva")
            self.detalle.set_text(
                f"Costo mínimo {ruta_definitiva.distancia:g} · "
                "el destino ya tiene su distancia confirmada"
            )
            self.camino_texto.set_text(
                "Ruta óptima: " + " → ".join(map(str, nodos_camino))
            )
            color_camino = MEJORA
            visitados = set(evento.visitados)
            actual = self.destino
        else:
            visitados = set(evento.visitados)
            actual = evento.nodo
            if evento.tipo == "inicio":
                nodos_camino, aristas_camino = [self.origen], []
                detalle = "Origen en 0; los demás nodos empiezan con distancia infinita"
                self.camino_texto.set_text(f"Comienza en el nodo {self.origen}")
            elif evento.tipo == "seleccion":
                camino = self.camino_hasta(evento.nodo, evento)
                nodos_camino, aristas_camino = camino.nodos, camino.aristas
                detalle = (f"Se confirma el nodo {evento.nodo} "
                           f"con distancia {evento.distancias[evento.nodo]:g}")
                self.camino_texto.set_text(
                    "Camino confirmado: " + " → ".join(map(str, nodos_camino))
                )
            else:
                camino = self.camino_hasta(evento.nodo, evento)
                nodos_camino = camino.nodos + [evento.vecino]
                aristas_camino = camino.aristas + [evento.clave]
                anterior = ("∞" if math.isinf(evento.anterior) else
                            f"{evento.anterior:g}")
                decision = "mejora" if evento.mejora else "no mejora"
                detalle = (f"Se evalúa {evento.nodo}–{evento.vecino} "
                           f"(peso {evento.peso:g}): propuesto {evento.propuesta:g}, "
                           f"anterior {anterior} · {decision}")
                self.camino_texto.set_text(
                    "Ruta examinada: " + " → ".join(map(str, nodos_camino))
                )
            self.titulo.set_text(
                f"Dijkstra {self.origen} → {self.destino} · búsqueda {indice:03d}/{total:03d}"
            )
            self.detalle.set_text(detalle)
            color_camino = (DESCARTADA if evento.tipo == "evaluacion" and not evento.mejora
                            else MEJORA)

        for clave in aristas_camino:
            a, b = self.extremos[clave]
            trazo = "--" if (evento.tipo == "evaluacion" and
                              not evento.mejora and clave == evento.clave and
                              ruta_definitiva is None) else "-"
            self.dibujar_arista(a, b, color_camino, 5.0, 5,
                                self.curvas.get(clave, 0), trazo)
            self.pesos[clave].set_color(color_camino)

        colores = []
        for nodo in self.nodos:
            if nodo in visitados:
                color = "#168369"
            elif not math.isinf(evento.distancias[nodo]):
                color = "#e4a846"
            else:
                color = "#356ca1"
            if nodo == actual and evento.tipo != "inicio":
                color = "#e88436"
            if evento.tipo == "evaluacion" and nodo == evento.vecino:
                color = color_camino
            if ruta_definitiva is not None and nodo in nodos_camino:
                color = "#e4a846"
            if nodo == self.origen:
                color = "#168369"
            if ruta_definitiva is not None and nodo == self.destino:
                color = "#b6533d"
            colores.append(color)
        self.marca_nodos.set_facecolors(colores)

        for nodo, etiqueta in self.tabla.items():
            distancia = evento.distancias[nodo]
            dist = "∞" if math.isinf(distancia) else f"{distancia:g}"
            ant = evento.antecesores[nodo][0] if nodo in evento.antecesores else "—"
            etiqueta.set_text(f"{nodo:>2}   d={dist:<4}  ant={ant}")
            etiqueta.set_color("#168369" if nodo in visitados else AZUL)

        self.figura.savefig(archivo, dpi=120, facecolor="white")


def nombre_evento(indice, evento):
    if evento.tipo == "inicio":
        return f"busqueda_{indice:03d}_inicio.png"
    if evento.tipo == "seleccion":
        return f"busqueda_{indice:03d}_seleccion_{evento.nodo}.png"
    decision = "mejora" if evento.mejora else "sin_mejora"
    return (f"busqueda_{indice:03d}_evalua_"
            f"{evento.nodo}_a_{evento.vecino}_{decision}.png")


def generar_capturas(origen, destino, raiz=RAIZ):
    grafo = construir_grafo(False)
    if origen not in grafo or destino not in grafo:
        raise ValueError("El origen y el destino deben existir en el grafo.")
    eventos = []
    ruta = dijkstra(grafo, origen, destino, registrar_evento=eventos.append)
    if ruta is None:
        raise ValueError(f"No hay ruta entre {origen} y {destino}.")

    carpeta = Path(raiz).resolve() / f"capturas_nodo_inicial_{origen}"
    if not carpeta.is_relative_to(Path(raiz).resolve()):
        raise ValueError("La carpeta de capturas debe estar dentro del proyecto.")
    carpeta.mkdir(parents=True, exist_ok=True)
    # Sustituye únicamente las imágenes generadas por versiones anteriores.
    for patron in ("paso_*.png", "busqueda_*.png", "ruta_definitiva_*.png"):
        for anterior in carpeta.glob(patron):
            if anterior.is_file() and anterior.resolve().parent == carpeta:
                anterior.unlink()

    dibujador = DibujadorBusqueda(grafo, origen, destino)
    archivos = []
    resumen = [f"Dijkstra: {origen} → {destino}",
               f"Decisiones de búsqueda: {len(eventos)}",
               "Ruta definitiva: " + " → ".join(map(str, ruta.nodos)),
               f"Costo mínimo: {ruta.distancia:g}", ""]
    for indice, evento in enumerate(eventos):
        archivo = carpeta / nombre_evento(indice, evento)
        dibujador.mostrar(evento, indice, len(eventos) - 1, archivo)
        archivos.append(archivo)
        if evento.tipo == "evaluacion":
            anterior = "∞" if math.isinf(evento.anterior) else f"{evento.anterior:g}"
            decision = "mejora" if evento.mejora else "sin mejora"
            resumen.append(
                f"{archivo.name}: {evento.nodo}–{evento.vecino}, peso {evento.peso:g}, "
                f"propuesto {evento.propuesta:g}, anterior {anterior}, {decision}"
            )
        elif evento.tipo == "seleccion":
            resumen.append(
                f"{archivo.name}: nodo {evento.nodo} confirmado, "
                f"distancia {evento.distancias[evento.nodo]:g}"
            )
        else:
            resumen.append(f"{archivo.name}: inicio en {origen}")

    definitiva = carpeta / f"ruta_definitiva_{origen}_a_{destino}.png"
    dibujador.mostrar(eventos[-1], len(eventos), len(eventos), definitiva, ruta)
    archivos.append(definitiva)
    resumen.extend(["", f"{definitiva.name}: costo {ruta.distancia:g}"])
    (carpeta / "resumen.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    dibujador.figura.clear()
    return archivos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origen", nargs="?", type=int)
    parser.add_argument("destino", nargs="?", type=int)
    argumentos = parser.parse_args()
    if (argumentos.origen is None) != (argumentos.destino is None):
        parser.error("Escribe ambos nodos o ninguno.")
    pares = ([(24, 8), (8, 24)] if argumentos.origen is None else
             [(argumentos.origen, argumentos.destino)])
    for origen, destino in pares:
        archivos = generar_capturas(origen, destino)
        print(f"{origen} -> {destino}: {len(archivos)} imagenes en {archivos[0].parent}")


if __name__ == "__main__":
    main()
