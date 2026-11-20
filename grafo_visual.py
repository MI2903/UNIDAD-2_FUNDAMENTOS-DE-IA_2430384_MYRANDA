"""Visualizador de los grafos de las fotografías y rutas mínimas.

NetworkX almacena el grafo; los tres algoritmos están implementados aquí.
"""

from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass
from html import escape

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette
from qdarkstyle.light.palette import LightPalette
from PyQt6.QtGui import QFont, QFontDatabase, QIntValidator
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QStatusBar, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)


# Posiciones aproximadas de las fotografías: (x, y), con y creciente hacia arriba.
POSICIONES = {
    29: (0.1, 6.4), 21: (0.1, 5.0), 24: (0.1, 3.4),
    7: (2.1, 6.4), 11: (1.6, 4.4), 14: (2.8, 4.4),
    10: (2.8, 2.9), 27: (2.5, 1.3), 6: (4.3, 5.1),
    9: (4.3, 4.4), 4: (4.4, 1.5), 18: (5.4, 1.5),
    66: (6.6, 0.3), 15: (6.9, 5.1), 8: (6.2, 4.1),
    12: (7.3, 2.8), 13: (8.5, 0.3), 0: (9.0, 6.1),
    1: (9.0, 4.1), 23: (9.5, 3.1), 16: (9.5, 1.4),
    44: (10.6, 6.6), 3: (10.6, 5.1), 2: (10.6, 4.1),
    17: (11.0, 2.4), 5: (12.4, 5.5), 69: (12.4, 4.1),
    70: (13.4, 2.0), 20: (14.2, 4.8), 25: (14.2, 3.4),
}

# (extremo A, extremo B, peso). Se conservan las dos aristas 12–16.
ARISTAS_POSITIVAS = [
    (29, 7, 5), (29, 21, 3), (21, 24, 2), (21, 11, 2),
    (24, 11, 7), (7, 11, 5), (7, 14, 3), (7, 6, 10),
    (7, 1, 20), (11, 14, 9), (11, 10, 7), (14, 10, 9),
    (14, 6, 5), (14, 9, 5), (14, 4, 8), (10, 9, 3),
    (10, 4, 6), (10, 27, 9), (10, 1, 5), (27, 4, 3),
    (27, 66, 1), (6, 9, 7), (6, 8, 4), (6, 15, 3),
    (4, 8, 15), (4, 18, 5), (18, 66, 6),
    (18, 12, 4), (15, 8, 2), (15, 0, 8), (15, 1, 8),
    (8, 1, 7), (8, 12, 9), (12, 1, 7), (12, 66, 7),
    (12, 13, 9), (12, 16, 16), (12, 16, 3),
    (66, 13, 777), (13, 16, 3), (0, 1, 2), (0, 44, 70),
    (1, 2, 3), (1, 23, 7), (23, 2, 3), (23, 16, 11),
    (44, 3, 80), (44, 5, 69), (3, 2, 9), (3, 5, 2),
    (3, 69, 6), (2, 69, 11), (2, 17, 5), (16, 17, 7),
    (17, 69, 4), (17, 70, 6), (5, 69, 3), (5, 20, 2),
    (69, 20, 3), (69, 25, 6), (69, 70, 8),
    (20, 25, 7), (25, 70, 17),
]

# Sobrescrituras manuscritas visibles en la segunda fotografía.
CAMBIOS_NEGATIVOS = {
    frozenset((21, 11)): -2,
    frozenset((24, 11)): -7,
    frozenset((7, 14)): -7,
    frozenset((6, 9)): -7,
    frozenset((10, 9)): -3,
    frozenset((23, 16)): -11,
    frozenset((3, 69)): -6,
    frozenset((5, 20)): -2,
    frozenset((66, 13)): -777,
}


def construir_grafo(negativo: bool) -> nx.MultiGraph | nx.MultiDiGraph:
    grafo = nx.MultiDiGraph() if negativo else nx.MultiGraph()
    grafo.add_nodes_from(POSICIONES)
    for indice, (a, b, peso) in enumerate(ARISTAS_POSITIVAS, start=1):
        if negativo:
            peso = CAMBIOS_NEGATIVOS.get(frozenset((a, b)), peso)
            # Orden total por posición: izquierda a derecha, luego arriba abajo.
            if (POSICIONES[a][0], -POSICIONES[a][1], a) > (
                POSICIONES[b][0], -POSICIONES[b][1], b
            ):
                a, b = b, a
        grafo.add_edge(a, b, key=f"e{indice:02d}", weight=peso)
    return grafo


def curvaturas_del_grafo(grafo):
    """Separa aristas paralelas y conexiones largas en el dibujo."""
    aristas = list(grafo.edges(keys=True))
    paralelas = [clave for a, b, clave in aristas if {a, b} == {12, 16}]
    curvaturas = {clave: (-0.20 if i == 0 else 0.20)
                   for i, clave in enumerate(paralelas)}
    curvaturas.update({"e09": -0.28, "e19": -0.17, "e21": -0.16})
    return curvaturas


@dataclass
class Ruta:
    distancia: float
    nodos: list[int]
    aristas: list[str]


@dataclass
class PasoDijkstra:
    nodo: int
    distancia: float
    visitados: tuple[int, ...]
    camino: Ruta


@dataclass
class EventoDijkstra:
    """Estado completo tras una decisión de la búsqueda de Dijkstra."""
    tipo: str
    nodo: int | None
    vecino: int | None
    clave: str | None
    peso: float | None
    propuesta: float | None
    anterior: float | None
    mejora: bool | None
    visitados: tuple[int, ...]
    distancias: dict[int, float]
    antecesores: dict[int, tuple[int, str]]


def vecinos(grafo, nodo):
    """Devuelve aristas salientes; MultiGraph permite aquí ambos sentidos."""
    for _, destino, clave, datos in grafo.edges(nodo, keys=True, data=True):
        yield destino, clave, datos["weight"]


def arcos(grafo):
    for origen in grafo.nodes:
        for destino, clave, peso in vecinos(grafo, origen):
            yield origen, destino, clave, peso


def reconstruir(origen, destino, anteriores, distancia):
    if math.isinf(distancia):
        return None
    if origen == destino:
        return Ruta(0, [origen], [])
    nodos, claves = [destino], []
    actual = destino
    while actual != origen:
        if actual not in anteriores:
            return None
        anterior, clave = anteriores[actual]
        claves.append(clave)
        nodos.append(anterior)
        actual = anterior
    nodos.reverse()
    claves.reverse()
    return Ruta(distancia, nodos, claves)


def dijkstra(grafo, origen, destino, registrar=None, registrar_evento=None):
    if any(datos["weight"] < 0 for _, _, _, datos in grafo.edges(keys=True, data=True)):
        raise ValueError("Dijkstra no admite pesos negativos.")
    distancias = {n: math.inf for n in grafo.nodes}
    distancias[origen] = 0
    anteriores = {}
    cola = [(0, origen)]
    visitados = []
    visitados_set = set()

    def registrar_estado(tipo, nodo=None, vecino=None, clave=None, peso=None,
                         propuesta=None, anterior=None, mejora=None):
        if registrar_evento is not None:
            registrar_evento(EventoDijkstra(
                tipo, nodo, vecino, clave, peso, propuesta, anterior, mejora,
                tuple(visitados), distancias.copy(), anteriores.copy(),
            ))

    registrar_estado("inicio", origen)
    while cola:
        distancia, nodo = heapq.heappop(cola)
        if distancia != distancias[nodo]:
            continue
        visitados.append(nodo)
        visitados_set.add(nodo)
        registrar_estado("seleccion", nodo)
        if nodo != destino:
            for vecino, clave, peso in vecinos(grafo, nodo):
                if vecino in visitados_set:
                    continue
                anterior = distancias[vecino]
                nueva = distancia + peso
                mejora = nueva < anterior
                if mejora:
                    distancias[vecino] = nueva
                    anteriores[vecino] = (nodo, clave)
                    heapq.heappush(cola, (nueva, vecino))
                registrar_estado("evaluacion", nodo, vecino, clave, peso,
                                 nueva, anterior, mejora)
        if registrar is not None:
            registrar(PasoDijkstra(
                nodo, distancia, tuple(visitados),
                reconstruir(origen, nodo, anteriores, distancia),
            ))
        if nodo == destino:
            break
    return reconstruir(origen, destino, anteriores, distancias[destino])


def bellman_ford(grafo, origen, destino):
    distancias = {n: math.inf for n in grafo.nodes}
    distancias[origen] = 0
    anteriores = {}
    lista_arcos = list(arcos(grafo))
    for _ in range(len(grafo.nodes) - 1):
        cambio = False
        for a, b, clave, peso in lista_arcos:
            if distancias[a] != math.inf and distancias[a] + peso < distancias[b]:
                distancias[b] = distancias[a] + peso
                anteriores[b] = (a, clave)
                cambio = True
        if not cambio:
            break
    for a, b, _, peso in lista_arcos:
        if distancias[a] != math.inf and distancias[a] + peso < distancias[b]:
            raise ValueError("Hay un ciclo negativo alcanzable desde el origen.")
    return reconstruir(origen, destino, anteriores, distancias[destino])


def floyd_warshall(grafo, origen, destino):
    nodos = list(grafo.nodes)
    distancia = {i: {j: (0 if i == j else math.inf) for j in nodos} for i in nodos}
    siguiente = {i: {} for i in nodos}
    for a, b, clave, peso in arcos(grafo):
        if peso < distancia[a][b]:
            distancia[a][b] = peso
            siguiente[a][b] = (b, clave)
    for k in nodos:
        for i in nodos:
            if distancia[i][k] == math.inf:
                continue
            for j in nodos:
                nueva = distancia[i][k] + distancia[k][j]
                if nueva < distancia[i][j]:
                    distancia[i][j] = nueva
                    siguiente[i][j] = siguiente[i][k]
    if any(distancia[n][n] < 0 for n in nodos):
        raise ValueError("El grafo contiene un ciclo negativo.")
    if distancia[origen][destino] == math.inf:
        return None
    if origen == destino:
        return Ruta(0, [origen], [])
    camino, claves = [origen], []
    actual = origen
    while actual != destino:
        actual, clave = siguiente[actual][destino]
        camino.append(actual)
        claves.append(clave)
    return Ruta(distancia[origen][destino], camino, claves)


def matriz_adyacencia(grafo):
    nodos = sorted(grafo.nodes)
    matriz = [[0 if a == b else math.inf for b in nodos] for a in nodos]
    indices = {n: i for i, n in enumerate(nodos)}
    for a, b, _, peso in arcos(grafo):
        i, j = indices[a], indices[b]
        matriz[i][j] = min(matriz[i][j], peso)
    return nodos, matriz


def matriz_incidencia(grafo):
    nodos = sorted(grafo.nodes)
    aristas = sorted(grafo.edges(keys=True, data=True), key=lambda arista: int(arista[2][1:]))
    matriz = [[0] * len(aristas) for _ in nodos]
    indices = {n: i for i, n in enumerate(nodos)}
    for columna, (a, b, _, _) in enumerate(aristas):
        matriz[indices[a]][columna] = -1 if grafo.is_directed() else 1
        matriz[indices[b]][columna] = 1
    return nodos, aristas, matriz


def texto_adyacencia(grafo):
    nodos, matriz = matriz_adyacencia(grafo)
    ancho = 6
    lineas = ["MATRIZ DE ADYACENCIA (pesos; ∞ = sin arista)", ""]
    lineas.append("".join(f"{x:>{ancho}}" for x in ["", *nodos]))
    for nodo, fila in zip(nodos, matriz):
        lineas.append("".join(f"{('∞' if math.isinf(x) else x):>{ancho}}" for x in [nodo, *fila]))
    return "\n".join(lineas)


def texto_incidencia(grafo):
    nodos, aristas, matriz = matriz_incidencia(grafo)
    ancho = 5
    lineas = ["MATRIZ DE INCIDENCIA", ""]
    lineas.append("".join(f"{x:>{ancho}}" for x in ["", *[e[2] for e in aristas]]))
    for nodo, fila in zip(nodos, matriz):
        lineas.append("".join(f"{x:>{ancho}}" for x in [nodo, *fila]))
    lineas.extend(["", "COLUMNAS (origen → destino en el grafo negativo):"])
    for a, b, clave, datos in aristas:
        conexion = "→" if grafo.is_directed() else "—"
        lineas.append(f"{clave}: {a} {conexion} {b}, peso = {datos['weight']}")
    return "\n".join(lineas)


class Aplicacion(QMainWindow):
    def __init__(self):
        super().__init__()
        # Qt puede iniciarse sin fuentes del sistema en algunos equipos.
        normal_id = QFontDatabase.addApplicationFont(findfont("DejaVu Sans"))
        mono_id = QFontDatabase.addApplicationFont(findfont("DejaVu Sans Mono"))
        normal = QFontDatabase.applicationFontFamilies(normal_id)
        mono = QFontDatabase.applicationFontFamilies(mono_id)
        QApplication.instance().setFont(QFont(normal[0] if normal else "Arial", 10))
        fuente_mono = mono[0] if mono else "Consolas"
        self.setWindowTitle("Grafos | Dijkstra · Bellman-Ford · Floyd-Warshall")
        self.resize(1800, 1000)
        self.setMinimumSize(1050, 650)
        self.grafo = construir_grafo(True)
        self.resultados = {}
        self.errores = {}
        self.nodos_calculados = None
        self.pasos_dijkstra = []
        self.animando_dijkstra = False
        self.artistas_animacion = []
        self.indice_animacion = 0
        self.herramienta_activa = None
        self.vista_ampliada = False
        self.colores = {}

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        principal = QVBoxLayout(central)
        principal.setContentsMargins(20, 18, 20, 12)
        principal.setSpacing(14)

        self.titulo = QLabel("Visualizador de rutas en grafos")
        self.titulo.setObjectName("titulo")
        principal.addWidget(self.titulo)

        barra = QFrame()
        barra.setObjectName("controles")
        self.barra_controles = barra
        controles = QHBoxLayout(barra)
        controles.setContentsMargins(22, 18, 22, 18)
        controles.setSpacing(18)
        self.selector_grafo = QComboBox()
        self.selector_grafo.addItems(["Positivos", "Negativos"])
        self.selector_grafo.setCurrentText("Negativos")
        self.selector_grafo.setMinimumWidth(130)
        self.agregar_control(controles, "Grafo", self.selector_grafo)
        self.entrada_origen = QLineEdit("0")
        self.entrada_origen.setValidator(QIntValidator(0, 99999))
        self.entrada_origen.setPlaceholderText("Inicio")
        self.entrada_origen.setFixedWidth(100)
        self.agregar_control(controles, "Nodo inicial", self.entrada_origen)
        self.entrada_destino = QLineEdit("0")
        self.entrada_destino.setValidator(QIntValidator(0, 99999))
        self.entrada_destino.setPlaceholderText("Final")
        self.entrada_destino.setFixedWidth(100)
        self.agregar_control(controles, "Nodo final", self.entrada_destino)
        boton = QPushButton("Calcular rutas")
        boton.setObjectName("calcular")
        controles.addWidget(boton, alignment=Qt.AlignmentFlag.AlignBottom)
        controles.addStretch()
        self.selector_ruta = QComboBox()
        self.selector_ruta.addItems(["Dijkstra", "Bellman-Ford", "Floyd-Warshall"])
        self.selector_ruta.setCurrentText("Bellman-Ford")
        self.selector_ruta.setMinimumWidth(165)
        self.agregar_control(controles, "Ruta destacada", self.selector_ruta)
        self.selector_tema = QComboBox()
        self.selector_tema.addItems(["Claro", "Oscuro"])
        self.selector_tema.setMinimumWidth(105)
        self.agregar_control(controles, "Tema", self.selector_tema)
        principal.addWidget(barra)

        self.pestanas = QTabWidget()
        principal.addWidget(self.pestanas, 1)
        pagina_grafo = QWidget()
        pagina_layout = QVBoxLayout(pagina_grafo)
        pagina_layout.setContentsMargins(10, 10, 10, 10)
        pagina_layout.setSpacing(12)

        panel_grafico = QFrame()
        panel_grafico.setObjectName("panel_grafico")
        grafico_layout = QVBoxLayout(panel_grafico)
        grafico_layout.setContentsMargins(14, 12, 14, 12)
        grafico_layout.setSpacing(8)
        herramientas = QHBoxLayout()
        herramientas.setSpacing(8)
        self.etiqueta_grafico = QLabel("Grafo interactivo")
        self.etiqueta_grafico.setObjectName("seccion")
        herramientas.addWidget(self.etiqueta_grafico)
        herramientas.addStretch()
        herramientas.addWidget(QLabel("Pesos"))
        self.selector_pesos = QComboBox()
        self.selector_pesos.addItems(["Ruta", "Todos"])
        self.selector_pesos.setToolTip("Muestra los pesos de la ruta elegida o de todas las aristas")
        herramientas.addWidget(self.selector_pesos)
        self.boton_zoom = self.boton_herramienta("Zoom", comprobable=True)
        self.boton_mover = self.boton_herramienta("Mover", comprobable=True)
        self.boton_restablecer = self.boton_herramienta("Restablecer")
        self.boton_guardar = self.boton_herramienta("Guardar imagen")
        self.boton_animar = self.boton_herramienta("Animar Dijkstra")
        self.boton_animar.setEnabled(False)
        self.boton_expandir = self.boton_herramienta("Ampliar grafo")
        for boton_herramienta in (self.boton_zoom, self.boton_mover,
                                  self.boton_restablecer, self.boton_guardar,
                                  self.boton_animar,
                                  self.boton_expandir):
            herramientas.addWidget(boton_herramienta)
        grafico_layout.addLayout(herramientas)
        figura = Figure(figsize=(14, 7), dpi=100, facecolor="#f8fafc")
        figura.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.045)
        self.ejes = figura.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(figura)
        grafico_layout.addWidget(self.canvas, 1)
        pagina_layout.addWidget(panel_grafico, 1)
        self.navegacion = NavigationToolbar2QT(self.canvas, self)
        self.navegacion.hide()
        self.temporizador_dibujo = QTimer(self)
        self.temporizador_dibujo.setSingleShot(True)
        self.temporizador_dibujo.timeout.connect(self.dibujar)
        self.canvas.mpl_connect("resize_event", lambda _: self.temporizador_dibujo.start(80))
        self.temporizador_animacion = QTimer(self)
        self.temporizador_animacion.setInterval(380)
        self.temporizador_animacion.timeout.connect(self.avanzar_animacion)

        panel_resultados = QFrame()
        panel_resultados.setObjectName("panel_resultados")
        self.panel_resultados = panel_resultados
        resultados_layout = QVBoxLayout(panel_resultados)
        resultados_layout.setContentsMargins(14, 12, 14, 12)
        resultados_layout.setSpacing(8)
        etiqueta_resultados = QLabel("Resultados")
        etiqueta_resultados.setObjectName("seccion")
        resultados_layout.addWidget(etiqueta_resultados)
        self.salida = QTextEdit()
        self.salida.setReadOnly(True)
        self.salida.setFixedHeight(124)
        self.salida.setObjectName("resultado_texto")
        resultados_layout.addWidget(self.salida)
        pagina_layout.addWidget(panel_resultados)
        self.pestanas.addTab(pagina_grafo, "Grafo y rutas")

        self.adyacencia = self.crear_pagina_matriz("Matriz de adyacencia", fuente_mono)
        self.incidencia = self.crear_pagina_matriz("Matriz de incidencia", fuente_mono)
        self.setStatusBar(QStatusBar())
        self.info_grafo = QLabel()
        self.statusBar().addWidget(self.info_grafo, 1)
        self.selector_grafo.currentTextChanged.connect(self.cambiar_grafo)
        self.selector_ruta.currentTextChanged.connect(self.cambiar_ruta_destacada)
        self.selector_pesos.currentTextChanged.connect(self.dibujar)
        self.selector_tema.currentTextChanged.connect(self.aplicar_tema)
        boton.clicked.connect(self.calcular)
        self.boton_zoom.clicked.connect(lambda _: self.alternar_herramienta("zoom"))
        self.boton_mover.clicked.connect(lambda _: self.alternar_herramienta("pan"))
        self.boton_restablecer.clicked.connect(self.restablecer_vista)
        self.boton_guardar.clicked.connect(self.guardar_imagen)
        self.boton_animar.clicked.connect(self.repetir_animacion_dijkstra)
        self.boton_expandir.clicked.connect(self.alternar_vista_ampliada)
        self.entrada_origen.returnPressed.connect(self.calcular)
        self.entrada_destino.returnPressed.connect(self.calcular)
        self.actualizar_matrices()
        self.mostrar_espera()
        self.actualizar_info_grafo()
        self.aplicar_tema()

    @staticmethod
    def agregar_control(contenedor, etiqueta, widget):
        columna = QVBoxLayout()
        columna.setSpacing(6)
        columna.addWidget(QLabel(etiqueta))
        columna.addWidget(widget)
        contenedor.addLayout(columna)

    @staticmethod
    def boton_herramienta(texto, comprobable=False):
        boton = QPushButton(texto)
        boton.setObjectName("accion_grafico")
        boton.setCheckable(comprobable)
        return boton

    def showEvent(self, evento):
        super().showEvent(evento)
        if hasattr(self, "temporizador_dibujo"):
            QTimer.singleShot(0, self.dibujar)

    def keyPressEvent(self, evento):
        if self.vista_ampliada and evento.key() == Qt.Key.Key_Escape:
            self.alternar_vista_ampliada()
            return
        super().keyPressEvent(evento)

    def alternar_vista_ampliada(self):
        if not self.vista_ampliada:
            self.geometria_anterior = self.saveGeometry()
            self.antes_maximizada = self.isMaximized()
            self.pesos_anteriores = self.selector_pesos.currentText()
            self.selector_pesos.blockSignals(True)
            self.selector_pesos.setCurrentText("Todos")
            self.selector_pesos.blockSignals(False)
            self.vista_ampliada = True
            self.titulo.hide()
            self.barra_controles.hide()
            self.panel_resultados.hide()
            self.pestanas.tabBar().hide()
            self.statusBar().hide()
            self.boton_expandir.setText("Volver a la interfaz")
            self.showMaximized()
        else:
            self.vista_ampliada = False
            self.titulo.show()
            self.barra_controles.show()
            self.panel_resultados.show()
            self.pestanas.tabBar().show()
            self.statusBar().show()
            self.boton_expandir.setText("Ampliar grafo")
            self.selector_pesos.blockSignals(True)
            self.selector_pesos.setCurrentText(self.pesos_anteriores)
            self.selector_pesos.blockSignals(False)
            if self.antes_maximizada:
                self.showMaximized()
            else:
                self.showNormal()
                self.restoreGeometry(self.geometria_anterior)
        QTimer.singleShot(0, self.dibujar)

    def aplicar_tema(self, _texto=None):
        oscuro = self.selector_tema.currentText() == "Oscuro"
        if oscuro:
            c = {
                "fondo": "#111b29", "superficie": "#1b2a3c", "lienzo": "#172438",
                "texto": "#edf4fc", "suave": "#b5c6d9", "borde": "#36506a",
                "pestana": "#22364c", "primario": "#3988d3", "hover": "#56a1e9",
                "activo": "#174f80", "arista": "#91a9c3", "negativo": "#f199b8",
                "peso": "#edf4fc", "etiqueta": "#26384d", "costo": "#79c5fa",
                "sin_ruta": "#ffb494", "estado": "#172438",
            }
            paleta = DarkPalette
        else:
            c = {
                "fondo": "#eef3f8", "superficie": "#ffffff", "lienzo": "#ffffff",
                "texto": "#173456", "suave": "#59718a", "borde": "#d4dfeb",
                "pestana": "#e6eef7", "primario": "#2367ad", "hover": "#174d87",
                "activo": "#dcecff", "arista": "#8297b1", "negativo": "#a04c73",
                "peso": "#233950", "etiqueta": "#ffffff", "costo": "#0e7490",
                "sin_ruta": "#ae4c39", "estado": "#e5edf6",
            }
            paleta = LightPalette
        self.colores = c
        base = qdarkstyle.load_stylesheet(qt_api="pyqt6", palette=paleta)
        personalizado = f"""
            QMainWindow, QWidget#central {{ background: {c['fondo']}; color: {c['texto']}; }}
            QLabel#titulo {{ font-size: 24px; font-weight: 700; color: {c['texto']}; }}
            QLabel#seccion {{ font-size: 14px; font-weight: 700; color: {c['texto']}; }}
            QFrame#controles, QFrame#panel_grafico, QFrame#panel_resultados {{
                background: {c['superficie']}; border: 1px solid {c['borde']};
                border-radius: 10px;
            }}
            QFrame#controles QLabel, QFrame#panel_grafico QLabel,
            QFrame#panel_resultados QLabel {{ background: transparent; border: 0; }}
            QComboBox, QLineEdit {{
                background: {c['superficie']}; color: {c['texto']};
                border: 1px solid {c['borde']}; border-radius: 6px;
                padding: 6px 9px; min-height: 28px;
            }}
            QComboBox:hover, QLineEdit:hover {{ border-color: {c['primario']}; }}
            QComboBox::drop-down {{
                border: 0; width: 25px; background: {c['pestana']};
                border-top-right-radius: 6px; border-bottom-right-radius: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {c['superficie']}; color: {c['texto']};
                selection-background-color: {c['primario']};
            }}
            QPushButton#calcular {{
                background: {c['primario']}; color: white; border: 0;
                border-radius: 6px; padding: 9px 18px; font-weight: 700;
                min-height: 28px;
            }}
            QPushButton#calcular:hover {{ background: {c['hover']}; }}
            QPushButton#accion_grafico {{
                background: {c['superficie']}; color: {c['texto']};
                border: 1px solid {c['borde']}; border-radius: 6px;
                padding: 6px 11px; min-height: 25px;
            }}
            QPushButton#accion_grafico:hover {{ border-color: {c['primario']}; color: {c['primario']}; }}
            QPushButton#accion_grafico:checked {{
                background: {c['activo']}; border-color: {c['primario']};
                color: {c['texto']};
            }}
            QTabWidget::pane {{ border: 1px solid {c['borde']}; background: {c['fondo']}; }}
            QTabBar::tab {{
                background: {c['pestana']}; color: {c['suave']};
                padding: 10px 18px; margin-right: 3px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {c['superficie']}; color: {c['primario']}; font-weight: 700;
            }}
            QTextEdit#resultado_texto, QPlainTextEdit {{
                background: {c['superficie']}; color: {c['texto']};
                border: 1px solid {c['borde']}; border-radius: 6px;
                padding: 8px;
            }}
            QStatusBar {{
                background: {c['estado']}; color: {c['suave']};
                border-top: 1px solid {c['borde']}; padding: 4px 12px;
            }}
            QStatusBar QLabel {{ color: {c['suave']}; background: transparent; }}
        """
        QApplication.instance().setStyleSheet(base + personalizado)
        self.canvas.figure.set_facecolor(c["lienzo"])
        if self.nodos_calculados is None:
            self.dibujar()
        else:
            self.mostrar_resultados()
            self.dibujar()

    def alternar_herramienta(self, modo):
        if self.herramienta_activa == modo:
            getattr(self.navegacion, modo)()
            self.herramienta_activa = None
        else:
            if self.herramienta_activa:
                getattr(self.navegacion, self.herramienta_activa)()
            getattr(self.navegacion, modo)()
            self.herramienta_activa = modo
        self.boton_zoom.setChecked(self.herramienta_activa == "zoom")
        self.boton_mover.setChecked(self.herramienta_activa == "pan")

    def restablecer_vista(self):
        if self.herramienta_activa:
            getattr(self.navegacion, self.herramienta_activa)()
            self.herramienta_activa = None
            self.boton_zoom.setChecked(False)
            self.boton_mover.setChecked(False)
        self.ejes.set_xlim(-0.55, 14.75)
        self.ejes.set_ylim(-0.25, 7.1)
        self.canvas.draw()

    def guardar_imagen(self):
        ruta, filtro = QFileDialog.getSaveFileName(
            self, "Guardar imagen del grafo", "grafo.png", "Imagen PNG (*.png);;Gráfico SVG (*.svg)"
        )
        if not ruta:
            return
        if not ruta.lower().endswith((".png", ".svg")):
            ruta += ".svg" if "SVG" in filtro else ".png"
        try:
            self.canvas.figure.savefig(ruta, dpi=180, bbox_inches="tight")
        except OSError as error:
            QMessageBox.warning(self, "No se pudo guardar", str(error))
            return
        self.statusBar().showMessage(f"Imagen guardada: {ruta}", 5000)

    def crear_pagina_matriz(self, titulo, fuente_mono):
        texto = QPlainTextEdit()
        texto.setReadOnly(True)
        texto.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        texto.setFont(QFont(fuente_mono, 10))
        self.pestanas.addTab(texto, titulo)
        return texto

    def actualizar_matrices(self):
        self.adyacencia.setPlainText(texto_adyacencia(self.grafo))
        self.incidencia.setPlainText(texto_incidencia(self.grafo))

    def cambiar_grafo(self, _texto=None):
        self.detener_animacion()
        negativo = self.selector_grafo.currentText() == "Negativos"
        self.grafo = construir_grafo(negativo)
        self.actualizar_matrices()
        self.resultados = {}
        self.errores = {}
        self.nodos_calculados = None
        self.pasos_dijkstra = []
        self.boton_animar.setEnabled(False)
        self.selector_ruta.blockSignals(True)
        self.selector_ruta.setCurrentText("Bellman-Ford" if negativo else "Dijkstra")
        self.selector_ruta.blockSignals(False)
        self.selector_pesos.blockSignals(True)
        self.selector_pesos.setCurrentText("Ruta" if negativo else "Todos")
        self.selector_pesos.blockSignals(False)
        self.mostrar_espera()
        self.actualizar_info_grafo()
        self.dibujar()

    def mostrar_espera(self):
        self.salida.setPlainText("Escribe los nodos de inicio y final y pulsa Calcular rutas.")

    def actualizar_info_grafo(self):
        self.info_grafo.setText(
            "Grafo negativo dirigido: flechas de izquierda a derecha y de arriba abajo. "
            "Dijkstra no admite pesos negativos."
            if self.grafo.is_directed() else
            "Grafo positivo no dirigido. Usa Zoom o Mover para explorar las aristas."
        )

    def cambiar_ruta_destacada(self, nombre):
        self.detener_animacion()
        if nombre == "Dijkstra" and self.pasos_dijkstra:
            self.iniciar_animacion_dijkstra()
        else:
            self.dibujar()

    def repetir_animacion_dijkstra(self):
        if not self.pasos_dijkstra:
            return
        if self.selector_ruta.currentText() != "Dijkstra":
            self.selector_ruta.setCurrentText("Dijkstra")
        else:
            self.iniciar_animacion_dijkstra()

    def detener_animacion(self):
        self.temporizador_animacion.stop()
        self.animando_dijkstra = False
        self.etiqueta_grafico.setText("Grafo interactivo")
        self.artistas_animacion = []

    def iniciar_animacion_dijkstra(self):
        if not self.pasos_dijkstra:
            return
        self.detener_animacion()
        self.selector_pesos.blockSignals(True)
        self.selector_pesos.setCurrentText("Todos")
        self.selector_pesos.blockSignals(False)
        self.animando_dijkstra = True
        self.indice_animacion = 0
        self.dibujar()
        self.avanzar_animacion()
        if self.animando_dijkstra:
            self.temporizador_animacion.start()

    def avanzar_animacion(self):
        ruta_final = self.resultados.get("Dijkstra")
        tramos_finales = len(ruta_final.aristas) if ruta_final else 0
        total = len(self.pasos_dijkstra) + tramos_finales
        if self.indice_animacion >= total:
            self.detener_animacion()
            self.dibujar()
            return

        for artista in self.artistas_animacion:
            try:
                artista.remove()
            except ValueError:
                pass
        self.artistas_animacion = []

        if self.indice_animacion < len(self.pasos_dijkstra):
            paso = self.pasos_dijkstra[self.indice_animacion]
            self.pintar_paso_dijkstra(
                paso.visitados, paso.camino.nodos, paso.camino.aristas,
                paso.nodo, final=False,
            )
            self.etiqueta_grafico.setText(
                f"Dijkstra {self.indice_animacion + 1}/{len(self.pasos_dijkstra)}"
                f" · nodo {paso.nodo} · costo {paso.distancia:g}"
            )
        else:
            tramo = self.indice_animacion - len(self.pasos_dijkstra) + 1
            self.pintar_paso_dijkstra(
                (), ruta_final.nodos[:tramo + 1], ruta_final.aristas[:tramo],
                ruta_final.nodos[tramo], final=True,
            )
            self.etiqueta_grafico.setText(
                f"Dijkstra · ruta final {tramo}/{tramos_finales}"
                f" · costo {ruta_final.distancia:g}"
            )
        self.indice_animacion += 1
        self.canvas.draw_idle()

    def pintar_paso_dijkstra(self, visitados, nodos_camino, aristas_camino,
                            nodo_actual, final):
        ax = self.ejes
        color_arista = "#e65134" if final else "#e7a548"
        for clave in aristas_camino:
            a, b = self.extremos_aristas[clave]
            parche = FancyArrowPatch(
                POSICIONES[a], POSICIONES[b], arrowstyle="-",
                connectionstyle=f"arc3,rad={self.curvaturas_aristas.get(clave, 0)}",
                linewidth=5.3, color=color_arista, shrinkA=19, shrinkB=19,
                capstyle="round", zorder=6,
            )
            ax.add_patch(parche)
            self.artistas_animacion.append(parche)

        colores_nodos = {n: "#4a9cb0" for n in visitados}
        colores_nodos.update({n: "#e7a548" for n in nodos_camino})
        if nodos_camino:
            colores_nodos[nodos_camino[0]] = "#178369"
        if final and nodos_camino and nodos_camino[-1] == self.nodos_calculados[1]:
            colores_nodos[nodos_camino[-1]] = "#b95637"
        else:
            colores_nodos[nodo_actual] = "#ee8f37"
        for nodo, color in colores_nodos.items():
            x, y = POSICIONES[nodo]
            marca = ax.scatter([x], [y], s=640, c=[color], edgecolors="white",
                               linewidths=1.6, zorder=7)
            texto = ax.text(x, y, str(nodo), ha="center", va="center",
                            color="white", fontsize=10, fontweight="bold", zorder=8)
            self.artistas_animacion.extend((marca, texto))

    def calcular(self):
        try:
            origen = int(self.entrada_origen.text().strip())
            destino = int(self.entrada_destino.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Nodos inválidos", "Escribe números enteros para inicio y final.")
            return
        if origen not in self.grafo or destino not in self.grafo:
            QMessageBox.warning(self, "Nodo inexistente", "El inicio y el final deben ser nodos del grafo.")
            return
        self.detener_animacion()
        self.nodos_calculados = (origen, destino)
        algoritmos = {"Dijkstra": dijkstra, "Bellman-Ford": bellman_ford,
                      "Floyd-Warshall": floyd_warshall}
        self.resultados = {}
        self.errores = {}
        self.pasos_dijkstra = []
        for nombre, funcion in algoritmos.items():
            try:
                if nombre == "Dijkstra":
                    self.resultados[nombre] = funcion(
                        self.grafo, origen, destino, self.pasos_dijkstra.append
                    )
                else:
                    self.resultados[nombre] = funcion(self.grafo, origen, destino)
            except ValueError as error:
                self.resultados[nombre] = None
                self.errores[nombre] = str(error)
        self.boton_animar.setEnabled(bool(self.pasos_dijkstra))
        self.mostrar_resultados()
        self.actualizar_info_grafo()
        if self.selector_ruta.currentText() == "Dijkstra" and self.pasos_dijkstra:
            self.iniciar_animacion_dijkstra()
        else:
            self.dibujar()

    def mostrar_resultados(self):
        origen, destino = self.nodos_calculados
        filas = []
        for nombre in ("Dijkstra", "Bellman-Ford", "Floyd-Warshall"):
            ruta = self.resultados.get(nombre)
            if nombre in self.errores:
                costo = f"<span style='color:{self.colores['sin_ruta']};'><b>No disponible</b></span>"
                detalle = self.errores[nombre]
            elif ruta is None:
                costo = f"<span style='color:{self.colores['sin_ruta']};'><b>Sin ruta</b></span>"
                detalle = f"No existe camino de {origen} a {destino}."
            else:
                costo = f"<span style='color:{self.colores['costo']};'>Costo: <b>{ruta.distancia:g}</b></span>"
                detalle = " → ".join(map(str, ruta.nodos))
            filas.append(
                f"<tr><td width='150'><b>{escape(nombre)}</b></td>"
                f"<td width='120'>{costo}</td><td>{escape(detalle)}</td></tr>"
            )
        self.salida.setHtml(
            f"<html><body style='color:{self.colores['texto']}; font-size:10pt;'>"
            f"<table width='100%' cellspacing='0' cellpadding='7'>{''.join(filas)}</table>"
            "</body></html>"
        )

    def dibujar(self):
        ax = self.ejes
        ax.clear()
        self.artistas_animacion = []
        ax.set_facecolor(self.colores["lienzo"])
        ruta = (None if self.animando_dijkstra else
                self.resultados.get(self.selector_ruta.currentText()))
        claves_ruta = set(ruta.aristas) if ruta else set()
        aristas = list(self.grafo.edges(keys=True, data=True))
        self.extremos_aristas = {clave: (a, b) for a, b, clave, _ in aristas}
        ax.set_xlim(-0.55, 14.75)
        ax.set_ylim(-0.25, 7.1)
        ax.set_aspect("equal" if self.vista_ampliada else 0.8, adjustable="box")
        ax.axis("off")
        # El borde blanco separa las aristas incluso cuando se cruzan.
        curvaturas = curvaturas_del_grafo(self.grafo)
        self.curvaturas_aristas = curvaturas
        # Las rutas elegidas se dibujan al final para mantenerlas visibles.
        orden_dibujo = sorted(aristas, key=lambda arista: arista[2] in claves_ruta)
        parches = {}
        for a, b, clave, datos in orden_dibujo:
            x1, y1 = POSICIONES[a]
            x2, y2 = POSICIONES[b]
            destacada = clave in claves_ruta
            curva = curvaturas.get(clave, 0)
            color = "#e65134" if destacada else (
                self.colores["negativo"] if datos["weight"] < 0 else self.colores["arista"]
            )
            estilo_flecha = "-|>" if self.grafo.is_directed() else "-"
            ancho = 4.2 if destacada else 2.2
            fondo_arista = FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle=estilo_flecha,
                connectionstyle=f"arc3,rad={curva}",
                mutation_scale=25 if destacada else 21,
                linewidth=ancho + 2.6, color=self.colores["lienzo"],
                shrinkA=18, shrinkB=18, capstyle="round",
                zorder=2 if destacada else 1,
            )
            ax.add_patch(fondo_arista)
            parche = FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=estilo_flecha,
                                     connectionstyle=f"arc3,rad={curva}",
                                     mutation_scale=23 if destacada else 19,
                                     linewidth=ancho,
                                     color=color, shrinkA=18, shrinkB=18,
                                     capstyle="round", zorder=2.1 if destacada else 1.1)
            ax.add_patch(parche)
            parches[clave] = parche
        origen, destino = self.nodos_calculados or (None, None)
        colores = []
        for n in self.grafo.nodes:
            if n == origen:
                colores.append("#178369")
            elif n == destino:
                colores.append("#b95637")
            elif ruta and n in ruta.nodos:
                colores.append("#e7a548")
            else:
                colores.append("#3669a7" if not self.grafo.is_directed() else "#536aa5")
        nodos_dibujados = nx.draw_networkx_nodes(
            self.grafo, POSICIONES, nodelist=list(self.grafo.nodes),
            node_color=colores, node_size=640, edgecolors="white",
            linewidths=1.6, ax=ax,
        )
        nodos_dibujados.set_zorder(3)
        etiquetas_nodos = nx.draw_networkx_labels(
            self.grafo, POSICIONES, font_size=10,
            font_weight="bold", font_color="white", ax=ax,
        )
        for etiqueta in etiquetas_nodos.values():
            etiqueta.set_zorder(4)
        # La primera pasada establece tamaños reales en píxeles. Las etiquetas
        # se colocan después con esos tamaños, evitando nodos y otras etiquetas.
        self.canvas.draw()
        aristas_visibles = (aristas if self.selector_pesos.currentText() == "Todos" else
                            [arista for arista in aristas if arista[2] in claves_ruta])
        ubicaciones = self.ubicar_pesos(aristas_visibles, aristas, parches, claves_ruta)
        for _, _, clave, datos in aristas_visibles:
            px, py = ubicaciones[clave]
            x, y = ax.transData.inverted().transform((px, py))
            ax.text(x, y, str(datos["weight"]), fontsize=10.5, fontweight="bold",
                    ha="center", va="center", zorder=5,
                    color=self.colores["negativo"] if datos["weight"] < 0 else self.colores["peso"],
                    bbox={"facecolor": self.colores["etiqueta"], "edgecolor": "none",
                          "alpha": 0.96, "boxstyle": "round,pad=0.22"})
        self.canvas.draw()

    @staticmethod
    def segmento_cruza_caja(p0, p1, caja):
        """Intersección de segmento y rectángulo mediante recorte paramétrico."""
        x0, y0 = p0
        dx, dy = p1[0] - x0, p1[1] - y0
        xmin, ymin, xmax, ymax = caja
        inicio, fin = 0.0, 1.0
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                     (-dy, y0 - ymin), (dy, ymax - y0)):
            if abs(p) < 1e-9:
                if q < 0:
                    return False
            else:
                fraccion = q / p
                if p < 0:
                    inicio = max(inicio, fraccion)
                else:
                    fin = min(fin, fraccion)
                if inicio > fin:
                    return False
        return True

    def ubicar_pesos(self, aristas_visibles, aristas, parches, claves_ruta):
        """Distribuye los pesos cerca del centro de cada arista en píxeles."""
        ax = self.ejes
        renderer = self.canvas.get_renderer()
        fuente = FontProperties(family="DejaVu Sans", weight="bold", size=10.5)
        penalidad_cruces = 25 if self.selector_pesos.currentText() == "Ruta" else 300
        nodos_px = [ax.transData.transform(POSICIONES[n]) for n in self.grafo.nodes]
        radio_nodo = math.sqrt(640 / math.pi) * self.canvas.figure.dpi / 72 + 5
        limites = ax.bbox
        curvas = {}
        segmentos = {}
        for _, _, clave, _ in aristas:
            vertices = parches[clave].get_path().transformed(parches[clave].get_transform()).vertices
            p0, p1, p2 = vertices[:3]
            curvas[clave] = (p0, p1, p2)
            puntos = []
            for paso in range(17):
                t = paso / 16
                puntos.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
                               (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]))
            tramos = []
            for a, b in zip(puntos, puntos[1:]):
                tramos.append((a, b, (min(a[0], b[0]), min(a[1], b[1]),
                                       max(a[0], b[0]), max(a[1], b[1]))))
            segmentos[clave] = (
                (min(p[0] for p in puntos), min(p[1] for p in puntos),
                 max(p[0] for p in puntos), max(p[1] for p in puntos)),
                tramos,
            )

        def cruza_nodo(caja):
            xmin, ymin, xmax, ymax = caja
            for nx_, ny_ in nodos_px:
                cercano_x = min(max(nx_, xmin), xmax)
                cercano_y = min(max(ny_, ymin), ymax)
                if (nx_ - cercano_x) ** 2 + (ny_ - cercano_y) ** 2 < radio_nodo ** 2:
                    return True
            return False

        def cruza_etiqueta(caja, anteriores):
            x1, y1, x2, y2 = caja
            return any(x1 < b[2] + 4 and x2 > b[0] - 4 and
                       y1 < b[3] + 4 and y2 > b[1] - 4 for b in anteriores)

        def evaluar(clave, texto, anteriores):
            ancho, alto, _ = renderer.get_text_width_height_descent(texto, fuente, ismath=False)
            ancho += 13
            alto += 13
            p0, p1, p2 = curvas[clave]

            def opcion(t, separacion):
                x_base = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
                y_base = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
                tx = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
                ty = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
                longitud = math.hypot(tx, ty) or 1
                normal_x, normal_y = -ty / longitud, tx / longitud
                x = x_base + normal_x * separacion
                y = y_base + normal_y * separacion
                caja = (x - ancho / 2, y - alto / 2,
                        x + ancho / 2, y + alto / 2)
                if (caja[0] < limites.x0 + 5 or caja[2] > limites.x1 - 5 or
                        caja[1] < limites.y0 + 5 or caja[3] > limites.y1 - 5):
                    return None
                if cruza_nodo(caja) or cruza_etiqueta(caja, anteriores):
                    return None
                caja_aristas = (caja[0] - 2, caja[1] - 2,
                                caja[2] + 2, caja[3] + 2)
                cruces = 0
                for otra, (envolvente, tramos) in segmentos.items():
                    if otra == clave or not cruza_etiqueta(caja_aristas, (envolvente,)):
                        continue
                    if any(cruza_etiqueta(caja_aristas, (envolvente_tramo,))
                           and self.segmento_cruza_caja(a, b, caja_aristas)
                           for a, b, envolvente_tramo in tramos):
                        cruces += 1
                puntuacion = cruces * penalidad_cruces + abs(separacion) * 1.8 + abs(t - 0.5) * 65
                return puntuacion, (x, y), caja

            mejor = None
            for t in (0.5, 0.43, 0.57, 0.35, 0.65, 0.27, 0.73, 0.2, 0.8):
                for separacion in (0, 13, -13, 24, -24, 36, -36,
                                   49, -49, 62, -62, 76, -76):
                    candidato = opcion(t, separacion)
                    if candidato is not None and (mejor is None or candidato[0] < mejor[0]):
                        mejor = candidato
                        if candidato[0] == 0:
                            return mejor
            if mejor is None:
                # La ventana estrecha puede dejar sin espacio a aristas cortas.
                # Amplía la búsqueda antes de permitir una superposición.
                for t in (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9):
                    for separacion in (90, -90, 110, -110, 135, -135):
                        candidato = opcion(t, separacion)
                        if candidato is not None and (mejor is None or candidato[0] < mejor[0]):
                            mejor = candidato
            if mejor is None:
                # En un tamaño de ventana muy reducido, conserva la etiqueta
                # junto al punto medio aunque no haya espacio libre completo.
                x = (p0[0] + 2 * p1[0] + p2[0]) / 4
                y = (p0[1] + 2 * p1[1] + p2[1]) / 4
                mejor = (float("inf"), (x, y),
                         (x - ancho / 2, y - alto / 2, x + ancho / 2, y + alto / 2))
            return mejor

        def prioridad(arista):
            _, _, clave, datos = arista
            p0, _, p2 = curvas[clave]
            largo = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
            return (largo, clave not in claves_ruta, datos["weight"] >= 0)

        colocadas = []
        posiciones = {}
        self._pesos_sin_espacio = []
        for _, _, clave, datos in sorted(aristas_visibles, key=prioridad):
            puntuacion, posicion, caja = evaluar(clave, str(datos["weight"]), colocadas)
            if math.isinf(puntuacion):
                self._pesos_sin_espacio.append(clave)
            posiciones[clave] = posicion
            colocadas.append(caja)
        self._cajas_pesos = colocadas
        return posiciones


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = Aplicacion()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
