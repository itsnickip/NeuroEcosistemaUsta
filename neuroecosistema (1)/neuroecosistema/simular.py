"""Simulación visual del NeuroEcosistema (pygame).

    uv run simular.py                         # escenario A, redes base entrenadas
    uv run simular.py --escenario C --presa reglas --depredador "red base"

Haz clic sobre un agente para ver sus entradas, las activaciones de sus capas
ocultas y las probabilidades de salida de la red (la acción que ejecuta es el argmax).

Controles:
  ESPACIO pausa · → un paso (en pausa) · +/- velocidad · 1/2/3 escenario A/B/C
  P cambia política de presas · D cambia política de depredadores
  R reinicia (misma semilla) · G nueva semilla · V radios de visión · ESC salir
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pygame

from neuroeco.config import DESCRIPCION_ESCENARIOS, ESCENARIOS
from neuroeco.mundo import ENTRADAS_DEPREDADOR, ENTRADAS_PRESA, NOMBRES_ACCIONES, Mundo
from neuroeco.politicas import (PoliticaAleatoria, PoliticaRed, PoliticaReglasDepredador,
                                PoliticaReglasPresa)
from neuroeco.red import RedMLP
from neuroeco.simulacion import avanzar_depr, avanzar_presas

ANCHO, ALTO, MUNDO_PX, PANEL_X = 1240, 800, 780, 800
C_FONDO, C_MUNDO, C_TINTA, C_TINTA2, C_GRILLA = (246, 245, 241), (252, 252, 251), (11, 11, 11), (82, 81, 78), (225, 224, 217)
C_PRESA, C_DEPR, C_PLANTA, C_OBST = (42, 120, 214), (227, 73, 72), (27, 175, 122), (150, 148, 140)
C_SEL, C_ACENTO = (237, 161, 0), (98, 80, 214)


def politicas_disponibles(dir_modelos: Path):
    pres, depr = {}, {}
    for arq, etiqueta in (("base", "red base"), ("pequena", "red pequeña")):
        for archivo, sufijo in (("", ""), ("_imitacion", " solo imitación"), ("_sin_entrenar", " sin entrenar")):
            if arq == "pequena" and sufijo:
                continue
            rp = dir_modelos / arq / f"presa{archivo}.npz"
            rd = dir_modelos / arq / f"depredador{archivo}.npz"
            if rp.exists():
                pres[etiqueta + sufijo] = PoliticaRed(RedMLP.cargar(rp), etiqueta + sufijo)
            if rd.exists():
                depr[etiqueta + sufijo] = PoliticaRed(RedMLP.cargar(rd), etiqueta + sufijo)
    pres.update({"reglas": PoliticaReglasPresa(), "aleatoria": PoliticaAleatoria()})
    depr.update({"reglas": PoliticaReglasDepredador(), "aleatoria": PoliticaAleatoria()})
    return pres, depr


class App:
    def __init__(self, args):
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("NeuroEcosistema — depredadores y presas")
        self.reloj = pygame.time.Clock()
        self.f = pygame.font.SysFont("dejavusans", 15)
        self.fp = pygame.font.SysFont("dejavusans", 12)
        self.fg = pygame.font.SysFont("dejavusans", 19, bold=True)
        self.pol_presas, self.pol_depr = politicas_disponibles(Path(args.modelos))
        self.n_presa = args.presa if args.presa in self.pol_presas else next(iter(self.pol_presas))
        self.n_depr = args.depredador if args.depredador in self.pol_depr else next(iter(self.pol_depr))
        self.escenario, self.semilla = args.escenario, args.semilla
        self.pausa, self.velocidad, self.ver_vision = False, 1, False
        self.sel = None  # ("presa"|"depredador", índice)
        self.reiniciar()

    # ------------------------------------------------------------ estado
    def reiniciar(self):
        self.mundo = Mundo(ESCENARIOS[self.escenario], self.semilla)
        self.pol_presas[self.n_presa].reiniciar(self.semilla)
        self.pol_depr[self.n_depr].reiniciar(self.semilla)
        self.escala = MUNDO_PX / max(self.mundo.cfg.ancho, self.mundo.cfg.alto)

    def avanzar(self):
        m = self.mundo
        if m.terminado:
            return
        m.paso_simulacion(avanzar_presas(m, self.pol_presas[self.n_presa]),
                          avanzar_depr(m, self.pol_depr[self.n_depr]))

    def a_px(self, p):
        return int(10 + p[0] * self.escala), int(10 + p[1] * self.escala)

    def seleccionar(self, pos_px):
        x = (pos_px[0] - 10) / self.escala
        y = (pos_px[1] - 10) / self.escala
        mejor, dmin = None, 3.0
        for esp, pos, viva in (("presa", self.mundo.presas_pos, self.mundo.presas_viva),
                               ("depredador", self.mundo.depr_pos, self.mundo.depr_viva)):
            for i, p in enumerate(pos):
                d = np.hypot(p[0] - x, p[1] - y)
                if viva[i] and d < dmin:
                    mejor, dmin = (esp, i), d
        self.sel = mejor

    def ciclar(self, dicc, actual):
        claves = list(dicc)
        return claves[(claves.index(actual) + 1) % len(claves)]

    # ------------------------------------------------------------ eventos
    def eventos(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < PANEL_X:
                self.seleccionar(e.pos)
            if e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    return False
                elif k == pygame.K_SPACE:
                    self.pausa = not self.pausa
                elif k == pygame.K_RIGHT and self.pausa:
                    self.avanzar()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.velocidad = min(20, self.velocidad + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.velocidad = max(1, self.velocidad - 1)
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.escenario = "ABC"[k - pygame.K_1]; self.sel = None; self.reiniciar()
                elif k == pygame.K_p:
                    self.n_presa = self.ciclar(self.pol_presas, self.n_presa); self.reiniciar()
                elif k == pygame.K_d:
                    self.n_depr = self.ciclar(self.pol_depr, self.n_depr); self.reiniciar()
                elif k == pygame.K_r:
                    self.sel = None; self.reiniciar()
                elif k == pygame.K_g:
                    self.semilla += 1; self.sel = None; self.reiniciar()
                elif k == pygame.K_v:
                    self.ver_vision = not self.ver_vision
        return True

    # ------------------------------------------------------------ dibujo
    def texto(self, s, x, y, fuente=None, color=C_TINTA):
        self.pantalla.blit((fuente or self.f).render(s, True, color), (x, y))

    def dibujar_mundo(self):
        m, c, esc = self.mundo, self.mundo.cfg, self.escala
        self.pantalla.set_clip(pygame.Rect(10, 10, c.ancho * esc + 1, c.alto * esc + 1))
        pygame.draw.rect(self.pantalla, C_MUNDO, (10, 10, c.ancho * esc, c.alto * esc))
        pygame.draw.rect(self.pantalla, C_GRILLA, (10, 10, c.ancho * esc, c.alto * esc), 1)
        for p, r in zip(m.obst_pos, m.obst_radio):
            pygame.draw.circle(self.pantalla, C_OBST, self.a_px(p), int(r * esc))
        for p, a in zip(m.plantas_pos, m.plantas_activa):
            if a:
                pygame.draw.circle(self.pantalla, C_PLANTA, self.a_px(p), 4)
        vis = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA) if self.ver_vision else None
        for esp, pos, viva, energia, color, rad, vision in (
                ("presa", m.presas_pos, m.presas_viva, m.presas_energia, C_PRESA, 6, c.vision_presa),
                ("depredador", m.depr_pos, m.depr_viva, m.depr_energia, C_DEPR, 9, c.vision_depredador)):
            for i, p in enumerate(pos):
                if not viva[i]:
                    continue
                q = self.a_px(p)
                if vis is not None:
                    pygame.draw.circle(vis, (*color, 25), q, int(vision * esc))
                pygame.draw.circle(self.pantalla, color, q, rad)
                pygame.draw.circle(self.pantalla, C_MUNDO, q, rad, 2)
                # barra de energía
                pygame.draw.rect(self.pantalla, C_GRILLA, (q[0] - 8, q[1] - rad - 6, 16, 3))
                pygame.draw.rect(self.pantalla, color, (q[0] - 8, q[1] - rad - 6, int(16 * energia[i]), 3))
                if self.sel == (esp, i):
                    pygame.draw.circle(self.pantalla, C_SEL, q, rad + 5, 2)
                    pygame.draw.circle(self.pantalla, C_SEL, q, int(vision * esc), 1)
        if vis is not None:
            self.pantalla.blit(vis, (0, 0))
        self.pantalla.set_clip(None)

    def dibujar_panel(self):
        m, x = self.mundo, PANEL_X + 10
        self.texto("NeuroEcosistema", x, 12, self.fg)
        self.texto(f"Escenario {self.escenario}: {DESCRIPCION_ESCENARIOS[self.escenario]}", x, 40, self.fp, C_TINTA2)
        self.texto(f"Semilla {self.semilla}   Paso {m.paso}/{m.cfg.max_pasos}   x{self.velocidad}"
                   f"{'   [PAUSA]' if self.pausa else ''}{'   [FIN]' if m.terminado else ''}", x, 58, self.fp, C_TINTA2)
        self.texto(f"Presas: {self.n_presa}", x, 82, self.f, C_PRESA)
        self.texto(f"Depredadores: {self.n_depr}", x, 100, self.f, C_DEPR)

        vivas = int(m.presas_viva.sum())
        e_p = m.presas_energia[m.presas_viva].mean() if vivas else 0
        e_d = m.depr_energia[m.depr_viva].mean() if m.depr_viva.any() else 0
        stats = [f"Presas vivas: {vivas}/{len(m.presas_viva)}   (hambre: {(m.presas_causa == 'hambre').sum()})",
                 f"Capturas: {sum(m.depr_capturas)}   Plantas comidas: {m.comida_total}",
                 f"Energía media  presas {e_p:.2f} · depredadores {e_d:.2f}",
                 f"Depredadores vivos: {int(m.depr_viva.sum())}/{len(m.depr_viva)}"]
        for i, s in enumerate(stats):
            self.texto(s, x, 128 + i * 18, self.fp)

        # mini gráfica de presas vivas
        gx, gy, gw, gh = x, 222, 410, 56
        pygame.draw.rect(self.pantalla, C_MUNDO, (gx, gy, gw, gh))
        h = m.historial_presas_vivas
        if len(h) > 1:
            n0 = len(m.presas_viva)
            pts = [(gx + int(gw * i / m.cfg.max_pasos), gy + 2 + int((gh - 4) * (1 - v / n0))) for i, v in enumerate(h)]
            pygame.draw.lines(self.pantalla, C_PRESA, False, pts, 2)
        self.texto("presas vivas en el tiempo", gx, gy - 16, self.fp, C_TINTA2)
        self.dibujar_inspector(x, 290)

        ayuda = ["Clic: inspeccionar agente · ESPACIO pausa · → paso",
                 "1/2/3 escenario · P/D política presa/depredador",
                 "R reiniciar · G nueva semilla · V visión · +/- velocidad"]
        for i, s in enumerate(ayuda):
            self.texto(s, x, ALTO - 62 + i * 18, self.fp, C_TINTA2)

    def dibujar_inspector(self, x, y):
        if self.sel is None:
            self.texto("Haz clic en un agente para ver su red.", x, y, self.f, C_TINTA2)
            return
        esp, i = self.sel
        m = self.mundo
        viva = m.presas_viva if esp == "presa" else m.depr_viva
        if not viva[i]:
            self.texto(f"{esp} #{i} ya no está viva.", x, y, self.f, C_TINTA2)
            return
        obs = (m.observar_presas() if esp == "presa" else m.observar_depredadores())[i]
        nombres = ENTRADAS_PRESA if esp == "presa" else ENTRADAS_DEPREDADOR
        pol = self.pol_presas[self.n_presa] if esp == "presa" else self.pol_depr[self.n_depr]
        color = C_PRESA if esp == "presa" else C_DEPR
        accion = (m.presas_accion if esp == "presa" else m.depr_accion)[i]
        self.texto(f"{esp.upper()} #{i}  ·  última acción: {NOMBRES_ACCIONES[accion]}", x, y, self.f, color)
        y += 24
        self.texto("Entradas", x, y, self.fp, C_TINTA2)
        for k, (n, v) in enumerate(zip(nombres, obs)):
            yy = y + 16 + k * 15
            self.texto(f"{n:>15s}", x - 6, yy, self.fp, C_TINTA2)
            cx = x + 150
            pygame.draw.line(self.pantalla, C_GRILLA, (cx, yy + 7), (cx + 100, yy + 7), 1)
            pygame.draw.line(self.pantalla, C_GRILLA, (cx + 50, yy), (cx + 50, yy + 14), 1)
            w = int(50 * np.clip(v, -1, 1))
            pygame.draw.rect(self.pantalla, color, (cx + 50 + min(0, w), yy + 3, abs(w), 9))
            self.texto(f"{v:+.2f}", cx + 106, yy, self.fp)

        if not isinstance(pol, PoliticaRed):
            self.texto(f"Política '{pol.nombre}': no es una red (sin activaciones).", x, y + 200, self.fp, C_TINTA2)
            return
        prob, capas = pol.red.adelante(obs, devolver_capas=True)
        prob = prob[0]
        # capas ocultas como tiras de color (azul negativo, naranja positivo)
        yh = y + 16 + len(nombres) * 15 + 8
        for li, act in enumerate(capas[1:-1]):
            self.texto(f"oculta {li + 1} ({act.shape[1]})", x, yh, self.fp, C_TINTA2)
            for j, a in enumerate(act[0]):
                a = float(np.clip(a, -1, 1))
                col = (235, 104, 52) if a > 0 else (42, 120, 214)
                mezcla = tuple(int(252 + (cc - 252) * abs(a)) for cc in col)
                pygame.draw.rect(self.pantalla, mezcla, (x + 100 + j * 18, yh, 16, 14))
            yh += 18
        self.texto("Salida softmax (acción = argmax)", x, yh + 4, self.fp, C_TINTA2)
        top = int(np.argmax(prob))
        for k, (n, p) in enumerate(zip(NOMBRES_ACCIONES, prob)):
            yy = yh + 22 + k * 15
            self.texto(f"{n:>7s}", x, yy, self.fp, C_TINTA if k == top else C_TINTA2)
            pygame.draw.rect(self.pantalla, C_GRILLA, (x + 70, yy + 3, 230, 9))
            pygame.draw.rect(self.pantalla, C_SEL if k == top else color, (x + 70, yy + 3, int(230 * p), 9))
            self.texto(f"{p * 100:5.1f}%", x + 306, yy, self.fp)

    def correr(self, max_frames=None):
        frames = 0
        while self.eventos():
            if not self.pausa:
                for _ in range(self.velocidad):
                    self.avanzar()
            self.pantalla.fill(C_FONDO)
            self.dibujar_mundo()
            self.dibujar_panel()
            pygame.display.flip()
            self.reloj.tick(30)
            frames += 1
            if max_frames and frames >= max_frames:
                break
        pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", default="A", choices=["A", "B", "C"])
    ap.add_argument("--semilla", type=int, default=1000)
    ap.add_argument("--modelos", default="modelos")
    ap.add_argument("--presa", default="red base")
    ap.add_argument("--depredador", default="red base")
    ap.add_argument("--captura", help="(pruebas) corre sin ventana N cuadros y guarda una imagen")
    a = ap.parse_args()
    if a.captura:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    app = App(a)
    if a.captura:
        n, ruta = a.captura.split(":")
        app.sel = ("presa", 0)
        for _ in range(int(n)):
            app.avanzar()
        if not app.mundo.presas_viva[0]:
            vivas = np.flatnonzero(app.mundo.presas_viva)
            app.sel = ("presa", int(vivas[0])) if len(vivas) else None
        app.pantalla.fill(C_FONDO); app.dibujar_mundo(); app.dibujar_panel()
        pygame.image.save(app.pantalla, ruta)
        pygame.quit()
        return
    app.correr()


if __name__ == "__main__":
    main()
