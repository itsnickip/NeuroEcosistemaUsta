"""Mundo 2D continuo con plantas, presas, depredadores y obstáculos.

Todo está vectorizado con NumPy: en cada paso se calcula una sola matriz de
distancias por pareja de especies y una sola pasada de la red por especie.
No depende de pygame, así que se usa igual para entrenar, evaluar y dibujar.
"""
from __future__ import annotations

import numpy as np

from .config import ConfigMundo

# ------------------------------------------------------------------ acciones
# 0 = quieto, 1..8 = ocho direcciones cada 45° (y crece hacia abajo, como en pantalla)
NOMBRES_ACCIONES = ("QUIETO", "E", "NE", "N", "NO", "O", "SO", "S", "SE")
N_ACCIONES = len(NOMBRES_ACCIONES)
_ang = np.deg2rad(np.arange(8) * 45.0)
DIRECCIONES = np.vstack([[0.0, 0.0], np.column_stack((np.cos(_ang), -np.sin(_ang)))])


def vector_a_accion(v: np.ndarray) -> np.ndarray:
    """Convierte vectores (n, 2) en la acción de dirección más cercana (0 si es nulo)."""
    v = np.atleast_2d(v)
    norma = np.linalg.norm(v, axis=1)
    ang = np.arctan2(-v[:, 1], v[:, 0])                      # deshace el eje y invertido
    k = np.round(ang / (np.pi / 4)).astype(int) % 8 + 1
    return np.where(norma < 1e-9, 0, k)


# --------------------------------------------------------------- entradas
ENTRADAS_PRESA = (
    "planta_dx", "planta_dy", "planta_dist",
    "depred_dx", "depred_dy", "depred_dist",
    "energia", "pos_x", "pos_y", "obst_dx", "obst_dy",
)
ENTRADAS_DEPREDADOR = (
    "presa_dx", "presa_dy", "presa_dist", "presas_visibles",
    "energia", "pos_x", "pos_y", "obst_dx", "obst_dy",
)


def _distancias(a: np.ndarray, b: np.ndarray):
    """Diferencias (na, nb, 2) y distancias (na, nb)."""
    d = b[None, :, :] - a[:, None, :]
    return d, np.sqrt((d ** 2).sum(axis=2))


def _mas_cercano(origen, destino, mascara_destino, vision):
    """Vector normalizado al objetivo visible más cercano.

    Devuelve (dx, dy, dist, n_visibles) normalizados por el radio de visión;
    si no hay nada a la vista: dx = dy = 0 y dist = 1.
    """
    n = len(origen)
    salida = np.zeros((n, 4))
    salida[:, 2] = 1.0
    if n == 0 or len(destino) == 0 or not mascara_destino.any():
        return salida
    diff, dist = _distancias(origen, destino)
    dist = np.where(mascara_destino[None, :], dist, np.inf)
    visibles = dist <= vision
    idx = dist.argmin(axis=1)
    dmin = dist[np.arange(n), idx]
    hay = dmin <= vision
    vec = diff[np.arange(n), idx] / vision
    salida[hay, 0:2] = vec[hay]
    salida[hay, 2] = dmin[hay] / vision
    salida[:, 3] = visibles.sum(axis=1)
    return salida


class Mundo:
    def __init__(self, cfg: ConfigMundo, semilla: int = 0):
        self.cfg = cfg
        self.semilla = semilla
        self.rng = np.random.default_rng(semilla)
        self.reiniciar()

    # ------------------------------------------------------------- creación
    def _punto_libre(self, n):
        """n posiciones aleatorias fuera de los obstáculos."""
        c = self.cfg
        pts = np.empty((n, 2))
        for i in range(n):
            for _ in range(200):
                p = self.rng.uniform([2, 2], [c.ancho - 2, c.alto - 2])
                if not self._dentro_obstaculo(p[None, :])[0]:
                    break
            pts[i] = p
        return pts

    def _dentro_obstaculo(self, pos):
        if len(self.obst_pos) == 0:
            return np.zeros(len(pos), dtype=bool)
        _, d = _distancias(pos, self.obst_pos)
        return (d < self.obst_radio[None, :]).any(axis=1)

    def reiniciar(self):
        c = self.cfg
        self.paso = 0
        self.obst_pos = np.empty((0, 2))
        self.obst_radio = np.empty(0)
        if c.n_obstaculos:
            self.obst_pos = self.rng.uniform([15, 15], [c.ancho - 15, c.alto - 15], (c.n_obstaculos, 2))
            self.obst_radio = self.rng.uniform(*c.radio_obstaculo, c.n_obstaculos)

        self.plantas_pos = self._punto_libre(c.n_plantas)
        self.plantas_activa = np.ones(c.n_plantas, dtype=bool)
        self.plantas_timer = np.zeros(c.n_plantas, dtype=int)

        self.presas_pos = self._punto_libre(c.n_presas)
        self.presas_energia = np.full(c.n_presas, c.energia_inicial_presa)
        self.presas_viva = np.ones(c.n_presas, dtype=bool)
        self.presas_pasos = np.zeros(c.n_presas, dtype=int)
        self.presas_comida = np.zeros(c.n_presas, dtype=int)
        self.presas_causa = np.array([""] * c.n_presas, dtype=object)
        self.presas_accion = np.zeros(c.n_presas, dtype=int)

        self.depr_pos = self._punto_libre(c.n_depredadores)
        self.depr_energia = np.full(c.n_depredadores, c.energia_inicial_depredador)
        self.depr_viva = np.ones(c.n_depredadores, dtype=bool)
        self.depr_pasos = np.zeros(c.n_depredadores, dtype=int)
        self.depr_capturas = np.zeros(c.n_depredadores, dtype=int)
        self.depr_accion = np.zeros(c.n_depredadores, dtype=int)
        self.depr_acercamiento = np.zeros(c.n_depredadores)  # suma de cercanía a la presa más próxima

        self.pasos_captura: list[int] = []
        self.comida_total = 0
        self.historial_presas_vivas = [int(self.presas_viva.sum())]
        self.terminado = False
        return self

    # ---------------------------------------------------------- percepción
    def _sensor_obstaculo(self, pos, vision):
        out = np.zeros((len(pos), 2))
        if len(self.obst_pos) == 0 or len(pos) == 0:
            return out
        diff, d = _distancias(pos, self.obst_pos)
        superficie = d - self.obst_radio[None, :]
        idx = superficie.argmin(axis=1)
        smin = superficie[np.arange(len(pos)), idx]
        cerca = smin < vision
        unit = diff[np.arange(len(pos)), idx] / np.maximum(d[np.arange(len(pos)), idx], 1e-9)[:, None]
        # magnitud crece al acercarse a la superficie (1 = tocando, 0 = en el borde de la visión)
        mag = np.clip(1 - smin / vision, 0, 1)
        out[cerca] = unit[cerca] * mag[cerca, None]
        return out

    def _pos_normalizada(self, pos):
        return np.column_stack((2 * pos[:, 0] / self.cfg.ancho - 1, 2 * pos[:, 1] / self.cfg.alto - 1))

    def observar_presas(self) -> np.ndarray:
        c = self.cfg
        planta = _mas_cercano(self.presas_pos, self.plantas_pos, self.plantas_activa, c.vision_presa)
        depr = _mas_cercano(self.presas_pos, self.depr_pos, self.depr_viva, c.vision_presa)
        return np.column_stack((
            planta[:, :3], depr[:, :3], self.presas_energia,
            self._pos_normalizada(self.presas_pos),
            self._sensor_obstaculo(self.presas_pos, c.vision_presa),
        ))

    def observar_depredadores(self) -> np.ndarray:
        c = self.cfg
        presa = _mas_cercano(self.depr_pos, self.presas_pos, self.presas_viva, c.vision_depredador)
        return np.column_stack((
            presa[:, :3], np.clip(presa[:, 3] / 5.0, 0, 1), self.depr_energia,
            self._pos_normalizada(self.depr_pos),
            self._sensor_obstaculo(self.depr_pos, c.vision_depredador),
        ))

    # ------------------------------------------------------------ dinámica
    def _mover(self, pos, viva, acciones, velocidad):
        paso = DIRECCIONES[acciones] * velocidad
        limite = np.array([self.cfg.ancho, self.cfg.alto])
        nuevo = np.clip(pos + paso, 0, limite)
        bloqueado = self._dentro_obstaculo(nuevo)
        if bloqueado.any():
            # deslizamiento: si el movimiento choca, intenta solo el eje x y luego solo el y
            for eje in (0, 1):
                parcial = pos[bloqueado].copy()
                parcial[:, eje] += paso[bloqueado, eje]
                parcial = np.clip(parcial, 0, limite)
                ok = ~self._dentro_obstaculo(parcial)
                idx = np.flatnonzero(bloqueado)[ok]
                nuevo[idx] = parcial[ok]
                bloqueado[idx] = False
            nuevo[bloqueado] = pos[bloqueado]
        nuevo[~viva] = pos[~viva]
        return nuevo

    def paso_simulacion(self, acc_presas, acc_depr):
        if self.terminado:
            return
        c = self.cfg
        acc_presas = np.asarray(acc_presas, dtype=int)
        acc_depr = np.asarray(acc_depr, dtype=int)
        self.presas_accion, self.depr_accion = acc_presas, acc_depr

        # 1. movimiento y gasto de energía
        self.presas_pos = self._mover(self.presas_pos, self.presas_viva, acc_presas, c.velocidad_presa)
        self.depr_pos = self._mover(self.depr_pos, self.depr_viva, acc_depr, c.velocidad_depredador)
        self.presas_energia -= np.where(acc_presas == 0, c.costo_quieto_presa, c.costo_mover_presa) * self.presas_viva
        self.depr_energia -= np.where(acc_depr == 0, c.costo_quieto_depredador, c.costo_mover_depredador) * self.depr_viva

        # 2. presas comen (cada planta la come la presa viva más cercana dentro del radio)
        if self.plantas_activa.any() and self.presas_viva.any():
            _, d = _distancias(self.plantas_pos, self.presas_pos)
            d[:, ~self.presas_viva] = np.inf
            d[~self.plantas_activa, :] = np.inf
            quien = d.argmin(axis=1)
            comidas = d[np.arange(len(d)), quien] <= c.radio_comer
            for planta in np.flatnonzero(comidas):
                presa = quien[planta]
                self.presas_energia[presa] = min(1.0, self.presas_energia[presa] + c.energia_planta)
                self.presas_comida[presa] += 1
                self.comida_total += 1
                self.plantas_activa[planta] = False
                self.plantas_timer[planta] = c.regeneracion_plantas

        # 2b. señal de acercamiento (solo se usa para el "shaping" de la aptitud del depredador)
        if self.depr_viva.any() and self.presas_viva.any():
            _, d = _distancias(self.depr_pos, self.presas_pos[self.presas_viva])
            cerca = np.clip(1 - d.min(axis=1) / c.vision_depredador, 0, 1)
            self.depr_acercamiento += cerca * self.depr_viva

        # 3. capturas (orden aleatorio de depredadores para no favorecer al índice 0)
        if self.depr_viva.any() and self.presas_viva.any():
            for k in self.rng.permutation(len(self.depr_pos)):
                if not self.depr_viva[k]:
                    continue
                d = np.linalg.norm(self.presas_pos - self.depr_pos[k], axis=1)
                d[~self.presas_viva] = np.inf
                j = int(d.argmin())
                if d[j] <= c.radio_captura:
                    self.presas_viva[j] = False
                    self.presas_causa[j] = "captura"
                    self.depr_energia[k] = min(1.0, self.depr_energia[k] + c.energia_captura)
                    self.depr_capturas[k] += 1
                    self.pasos_captura.append(self.paso + 1)

        # 4. muerte por hambre
        hambre = self.presas_viva & (self.presas_energia <= 0)
        self.presas_viva[hambre] = False
        self.presas_causa[hambre] = "hambre"
        self.depr_viva &= self.depr_energia > 0
        self.presas_energia = np.clip(self.presas_energia, 0, 1)
        self.depr_energia = np.clip(self.depr_energia, 0, 1)

        # 5. regeneración de plantas en un lugar nuevo
        self.plantas_timer[~self.plantas_activa] -= 1
        renacen = (~self.plantas_activa) & (self.plantas_timer <= 0)
        if renacen.any():
            self.plantas_pos[renacen] = self._punto_libre(int(renacen.sum()))
            self.plantas_activa[renacen] = True

        self.presas_pasos += self.presas_viva
        self.depr_pasos += self.depr_viva
        self.paso += 1
        self.historial_presas_vivas.append(int(self.presas_viva.sum()))
        # Fin: tiempo agotado o no quedan presas. Si mueren todos los depredadores el
        # episodio sigue, porque las presas todavía pueden morir de hambre.
        self.terminado = self.paso >= c.max_pasos or not self.presas_viva.any()

    # ------------------------------------------------------------ métricas
    def metricas(self) -> dict:
        c = self.cfg
        cap = np.array(self.pasos_captura, dtype=float)
        vivas = self.presas_viva
        return {
            "pasos": self.paso,
            "supervivencia_presas": float(self.presas_pasos.mean() / c.max_pasos),
            "presas_vivas_final": float(vivas.mean()),
            "comida_consumida": int(self.comida_total),
            "comida_por_presa": float(self.presas_comida.mean()),
            "capturas": int(self.depr_capturas.sum()),
            "capturas_por_depredador": float(self.depr_capturas.mean()) if c.n_depredadores else 0.0,
            "muertes_hambre_presas": int((self.presas_causa == "hambre").sum()),
            "energia_final_presas": float(self.presas_energia[vivas].mean()) if vivas.any() else 0.0,
            "energia_final_depredadores": float(self.depr_energia.mean()) if c.n_depredadores else 0.0,
            "depredadores_vivos_final": float(self.depr_viva.mean()) if c.n_depredadores else 0.0,
            "acercamiento_depredadores": float(self.depr_acercamiento.mean() / c.max_pasos) if c.n_depredadores else 0.0,
            "supervivencia_depredadores": float(self.depr_pasos.mean() / c.max_pasos) if c.n_depredadores else 0.0,
            "tiempo_primera_captura": float(cap.min()) if len(cap) else float("nan"),
            "tiempo_medio_captura": float(cap.mean()) if len(cap) else float("nan"),
        }
