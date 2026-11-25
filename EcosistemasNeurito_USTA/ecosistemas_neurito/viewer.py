"""Visor interactivo (Pygame) de EcosistemasNeurito.

Tema oscuro "noche en la sabana". El visor solo dibuja: el mundo, las políticas
y la inferencia de la red viven en módulos separados. Los botones visibles
replican los atajos de teclado.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np


# Paleta "noche en la sabana"
_BG = (15, 19, 30)
_PAPER = (25, 32, 48)
_PANEL_SOFT = (33, 42, 62)
_FIELD = (18, 36, 34)
_FIELD_DOT = (34, 62, 56)
_INK = (234, 238, 246)
_MUTED = (138, 151, 175)
_LINE = (48, 60, 84)
_PREY = (92, 205, 255)
_PREDATOR = (255, 104, 128)
_PLANT = (138, 222, 92)
_ACCENT = (255, 191, 72)
_DARK_TEXT = (20, 22, 30)

_ACTION_VECTORS = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

_INPUT_LABELS = {
    "energia": "Energía",
    "alimento_dx": "Alimento Δx",
    "alimento_dy": "Alimento Δy",
    "depredador_dx": "Depredador Δx",
    "depredador_dy": "Depredador Δy",
    "alimento_distancia": "Distancia alimento",
    "depredador_distancia": "Distancia depredador",
    "borde_distancia": "Distancia al borde",
    "presa_dx": "Presa Δx",
    "presa_dy": "Presa Δy",
    "presa_distancia": "Distancia presa",
    "posicion_x": "Posición x",
    "posicion_y": "Posición y",
    "presas_vivas": "Presas vivas (fracción)",
    "companero_distancia": "Distancia compañero",
}


def _load_model(value: Any) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "predict_proba"):
        return value
    from .network import RedNeurito

    return RedNeurito.load(Path(value))


def _normalise_policy(value: str, model: Any | None) -> str:
    text = value.strip().lower()
    if text in {"neural", "neuronal", "red"} and model is not None:
        return "neural"
    if text in {"random", "aleatoria", "aleatorio"}:
        return "random"
    return "rule"


def _policy_label(policy: str) -> str:
    return {"neural": "Neuronal", "rule": "Reglas", "random": "Azar"}[policy]


def run_viewer(
    scenario: str = "abundante",
    seed: int = 101,
    prey_model: Any | None = None,
    predator_model: Any | None = None,
    prey_policy: str = "neural",
    predator_policy: str = "neural",
    fps: int = 30,
) -> None:
    """Run an episode with clickable scenario, playback, policy and inspector controls.

    Models may be RedNeurito instances or .npz paths. Without a model, that species'
    neural option is unavailable and the rule policy is used instead.
    """
    import pygame

    from .policies import random_action, rule_action
    from .world import ACTIONS, PREDATOR_INPUTS, PREY_INPUTS, SCENARIOS, World

    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {scenario!r}")
    if fps <= 0:
        raise ValueError("fps debe ser positivo")

    models = {"prey": _load_model(prey_model), "predator": _load_model(predator_model)}
    policies = {
        "prey": _normalise_policy(prey_policy, models["prey"]),
        "predator": _normalise_policy(predator_policy, models["predator"]),
    }
    scenario_names = tuple(name for name in ("abundante", "escasez", "presion") if name in SCENARIOS)
    if len(scenario_names) != 3:
        scenario_names = tuple(SCENARIOS)
    current_scenario = scenario
    current_seed = int(seed)

    def new_episode() -> tuple[World, random.Random]:
        return World(SCENARIOS[current_scenario], current_seed), random.Random(current_seed + 8047)

    world, policy_rng = new_episode()
    selected: tuple[str, int] | None = ("prey", 0) if world.prey else None
    last_decisions: dict[tuple[str, int], dict[str, Any]] = {}
    inspector_tab = "decision"
    paused = False
    help_open = False
    steps_per_frame = 1

    pygame.init()
    pygame.display.set_caption("EcosistemasNeurito · USTA Tunja")
    screen = pygame.display.set_mode((1280, 800))
    clock = pygame.time.Clock()
    family = "verdana,segoeui,dejavusans,arial"
    font_title = pygame.font.SysFont(family, 25, bold=True)
    font_heading = pygame.font.SysFont(family, 16, bold=True)
    font_value = pygame.font.SysFont(family, 21, bold=True)
    font_body = pygame.font.SysFont(family, 14)
    font_small = pygame.font.SysFont(family, 12)
    font_tiny = pygame.font.SysFont(family, 11)

    world_rect = pygame.Rect(22, 92, 820, 646)
    side_rect = pygame.Rect(864, 92, 394, 646)
    scenario_buttons = {
        name: pygame.Rect(350 + index * 112, 34, 106, 35)
        for index, name in enumerate(scenario_names)
    }
    action_buttons = {
        "pause": pygame.Rect(904, 30, 82, 38),
        "restart": pygame.Rect(993, 30, 93, 38),
        "new": pygame.Rect(1093, 30, 105, 38),
        "help": pygame.Rect(1205, 30, 53, 38),
    }
    policy_buttons = {
        (species, option): pygame.Rect(884 + index * 116, y, 109, 31)
        for species, y in (("prey", 309), ("predator", 369))
        for index, option in enumerate(("neural", "rule", "random"))
    }
    tab_buttons = {
        "decision": pygame.Rect(884, 534, 173, 29),
        "inputs": pygame.Rect(1063, 534, 173, 29),
    }
    speed_buttons = {
        "slower": pygame.Rect(694, 702, 29, 27),
        "faster": pygame.Rect(793, 702, 29, 27),
    }
    help_close = pygame.Rect(895, 150, 54, 34)
    help_modal = pygame.Rect(295, 125, 690, 510)

    def draw_text(value: str, x: int, y: int, font: Any = None,
                  color: tuple[int, int, int] = _INK) -> None:
        screen.blit((font or font_body).render(str(value), True, color), (x, y))

    def draw_button(rect: Any, label: str, *, active: bool = False,
                    disabled: bool = False, primary: bool = False) -> None:
        hovered = rect.collidepoint(pygame.mouse.get_pos()) and not disabled and not help_open
        if disabled:
            fill, border, color = (21, 27, 40), _LINE, (82, 94, 116)
        elif active:
            fill, border, color = _ACCENT, _ACCENT, _DARK_TEXT
        elif primary:
            fill, border, color = ((255, 214, 130) if hovered else _ACCENT), _ACCENT, _DARK_TEXT
        elif hovered:
            fill, border, color = (44, 56, 82), _ACCENT, _INK
        else:
            fill, border, color = _PANEL_SOFT, _LINE, _INK
        pygame.draw.rect(screen, fill, rect, border_radius=rect.height // 2)
        pygame.draw.rect(screen, border, rect, 1, border_radius=rect.height // 2)
        rendered = font_small.render(label, True, color)
        screen.blit(rendered, rendered.get_rect(center=rect.center))

    def fmt_number(value: Any, digits: int = 1) -> str:
        if value is None:
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "—"

    def reset_episode(*, new_seed: bool = False) -> None:
        nonlocal current_seed, world, policy_rng, selected, paused, inspector_tab
        if new_seed:
            current_seed += 1
        world, policy_rng = new_episode()
        last_decisions.clear()
        selected = ("prey", 0) if world.prey else None
        inspector_tab = "decision"
        paused = False

    def set_policy(species: str, option: str) -> None:
        if option == "neural" and models[species] is None:
            return
        policies[species] = option
        for key in tuple(last_decisions):
            if key[0] == species:
                del last_decisions[key]

    def cycle_policy(species: str) -> None:
        options = ("neural", "rule", "random") if models[species] is not None else ("rule", "random")
        current = policies[species]
        set_policy(species, options[(options.index(current) + 1) % len(options)])

    def world_projection() -> tuple[float, float, float]:
        scale = min((world_rect.width - 28) / world.config.width,
                    (world_rect.height - 58) / world.config.height)
        x0 = world_rect.x + (world_rect.width - world.config.width * scale) / 2
        y0 = world_rect.y + 13 + (world_rect.height - 58 - world.config.height * scale) / 2
        return x0, y0, scale

    def point(x: float, y: float) -> tuple[int, int]:
        x0, y0, scale = world_projection()
        return int(x0 + x * scale), int(y0 + y * scale)

    def choose_agent(mouse_pos: tuple[int, int]) -> None:
        nonlocal selected, inspector_tab
        best: tuple[str, int] | None = None
        best_distance = 20.0
        for species, agents in (("prey", world.prey), ("predator", world.predators)):
            for index, agent in enumerate(agents):
                if not agent.alive:
                    continue
                distance = math.dist(mouse_pos, point(agent.x, agent.y))
                if distance < best_distance:
                    best, best_distance = (species, index), distance
        if best is not None:
            selected = best
            inspector_tab = "decision"

    def advance() -> None:
        actions_by_species: dict[str, list[int]] = {"prey": [], "predator": []}
        for species, agents in (("prey", world.prey), ("predator", world.predators)):
            for index, agent in enumerate(agents):
                if not agent.alive:
                    actions_by_species[species].append(0)
                    continue
                observation = np.asarray(world.observe(species, index), dtype=float)
                policy = policies[species]
                probabilities = None
                if policy == "neural":
                    probabilities = np.asarray(
                        models[species].predict_proba(observation.reshape(1, -1))[0],
                        dtype=float,
                    )
                    action = int(np.argmax(probabilities))
                elif policy == "random":
                    action = int(random_action(world, species, index, policy_rng))
                else:
                    action = int(rule_action(world, species, index))
                actions_by_species[species].append(action)
                last_decisions[(species, index)] = {
                    "observation": observation,
                    "action": action,
                    "probabilities": probabilities,
                }
        world.step(actions_by_species["prey"], actions_by_species["predator"])

    def draw_header() -> None:
        pygame.draw.circle(screen, _ACCENT, (34, 32), 9)
        pygame.draw.circle(screen, _BG, (34, 32), 4)
        draw_text("EcosistemasNeurito", 52, 16, font_title)
        draw_text("Red 8→14→16→12→5 · ReLU + Softmax · USTA Tunja",
                  24, 56, font_small, _MUTED)
        draw_text("ESCENARIO", 350, 12, font_tiny, _MUTED)
        for name, rect in scenario_buttons.items():
            label = {"abundante": "Abundancia", "escasez": "Escasez", "presion": "Presión"}.get(name, name)
            draw_button(rect, label, active=name == current_scenario)
        draw_text(f"Semilla #{current_seed}", 696, 43, font_small, _ACCENT)
        pause_label = "Finalizado" if world.done else "Continuar" if paused else "Pausar"
        draw_button(action_buttons["pause"], pause_label,
                    primary=not world.done, disabled=world.done)
        draw_button(action_buttons["restart"], "Reiniciar")
        draw_button(action_buttons["new"], "Nueva semilla")
        draw_button(action_buttons["help"], "?  Ayuda", active=help_open)

    def draw_world() -> None:
        pygame.draw.rect(screen, _PAPER, world_rect, border_radius=16)
        pygame.draw.rect(screen, _LINE, world_rect, 1, border_radius=16)
        x0, y0, scale = world_projection()
        playable = pygame.Rect(round(x0), round(y0), round(world.config.width * scale),
                               round(world.config.height * scale))
        pygame.draw.rect(screen, _FIELD, playable, border_radius=10)
        pygame.draw.rect(screen, (46, 92, 78), playable, 2, border_radius=10)

        # Rejilla de puntos en lugar de líneas.
        for gx in range(50, world.config.width, 50):
            for gy in range(50, world.config.height, 50):
                pygame.draw.circle(screen, _FIELD_DOT, point(gx, gy), 1)

        for plant in world.plants:
            px, py = point(plant.x, plant.y)
            pygame.draw.circle(screen, (40, 82, 46), (px, py), 9)
            pygame.draw.ellipse(screen, _PLANT, pygame.Rect(px - 6, py - 2, 12, 5))
            pygame.draw.ellipse(screen, _PLANT, pygame.Rect(px - 2, py - 6, 5, 12))

        for species, agents, color in (("prey", world.prey, _PREY),
                                       ("predator", world.predators, _PREDATOR)):
            for index, agent in enumerate(agents):
                if not agent.alive:
                    continue
                px, py = point(agent.x, agent.y)
                record = last_decisions.get((species, index))
                vx, vy = _ACTION_VECTORS.get(record["action"] if record else 0, (0, 0))
                if selected == (species, index):
                    pulse = 18 + round(3 * math.sin(pygame.time.get_ticks() / 200))
                    pygame.draw.circle(screen, _ACCENT, (px, py), pulse, 2)
                if species == "prey":
                    pygame.draw.circle(screen, (30, 70, 92), (px, py), 13)
                    pygame.draw.circle(screen, color, (px, py), 9)
                    pygame.draw.circle(screen, _BG, (px + 4 * vx, py + 4 * vy), 3)
                else:
                    pygame.draw.polygon(screen, color,
                                        ((px, py - 13), (px + 11, py),
                                         (px, py + 13), (px - 11, py)))
                    pygame.draw.polygon(screen, (120, 36, 52),
                                        ((px, py - 6), (px + 5, py),
                                         (px, py + 6), (px - 5, py)))
                    if vx or vy:
                        pygame.draw.line(screen, color, (px, py),
                                         (px + 19 * vx, py + 19 * vy), 2)
                bar = pygame.Rect(px - 14, py - 24, 28, 4)
                pygame.draw.rect(screen, (40, 50, 70), bar, border_radius=2)
                energy_width = round(bar.width * max(0.0, min(1.0, agent.energy)))
                if energy_width:
                    pygame.draw.rect(screen, color,
                                     pygame.Rect(bar.x, bar.y, energy_width, bar.height),
                                     border_radius=2)

        if paused or world.done:
            shade = pygame.Surface(playable.size, pygame.SRCALPHA)
            shade.fill((8, 10, 18, 140))
            screen.blit(shade, playable.topleft)
            banner = pygame.Rect(playable.centerx - 168, playable.centery - 47, 336, 94)
            pygame.draw.rect(screen, _PAPER, banner, border_radius=14)
            pygame.draw.rect(screen, _ACCENT, banner, 2, border_radius=14)
            title = "Episodio terminado" if world.done else "Simulación en pausa"
            detail = "Pulsa Reiniciar o Nueva semilla" if world.done else "Pulsa Continuar para seguir"
            heading = font_heading.render(title, True, _INK)
            subtitle = font_small.render(detail, True, _MUTED)
            screen.blit(heading, heading.get_rect(center=(banner.centerx, banner.y + 33)))
            screen.blit(subtitle, subtitle.get_rect(center=(banner.centerx, banner.y + 63)))

        footer_y = 710
        pygame.draw.circle(screen, _PLANT, (world_rect.x + 24, footer_y + 5), 5)
        draw_text("Plantas", world_rect.x + 36, footer_y - 3, font_small, _MUTED)
        pygame.draw.circle(screen, _PREY, (world_rect.x + 120, footer_y + 5), 6)
        draw_text("Presas", world_rect.x + 133, footer_y - 3, font_small, _MUTED)
        pygame.draw.polygon(screen, _PREDATOR,
                            ((world_rect.x + 222, footer_y - 2),
                             (world_rect.x + 228, footer_y + 5),
                             (world_rect.x + 222, footer_y + 12),
                             (world_rect.x + 216, footer_y + 5)))
        draw_text("Depredadores", world_rect.x + 237, footer_y - 3, font_small, _MUTED)
        draw_text("Clic en un agente para inspeccionarlo", world_rect.x + 379, footer_y - 3,
                  font_tiny, _MUTED)
        draw_button(speed_buttons["slower"], "−", disabled=steps_per_frame == 1)
        speed = font_small.render(f"{steps_per_frame}×", True, _INK)
        screen.blit(speed, speed.get_rect(center=(758, footer_y + 5)))
        draw_button(speed_buttons["faster"], "+", disabled=steps_per_frame == 16)

    def draw_sidebar() -> None:
        pygame.draw.rect(screen, _PAPER, side_rect, border_radius=16)
        pygame.draw.rect(screen, _LINE, side_rect, 1, border_radius=16)
        pygame.draw.rect(screen, _ACCENT, pygame.Rect(side_rect.x + 20, side_rect.y, 60, 3))
        left, right = side_rect.x + 20, side_rect.right - 20
        metrics = world.metrics()
        draw_text("Episodio", left, 106, font_heading)
        status = "FINALIZADO" if world.done else "EN PAUSA" if paused else "EN CURSO"
        status_color = _PREDATOR if world.done else _ACCENT if paused else _PLANT
        badge = pygame.Rect(right - 104, 105, 104, 27)
        pygame.draw.rect(screen, _PANEL_SOFT, badge, border_radius=13)
        label = font_tiny.render(status, True, status_color)
        screen.blit(label, label.get_rect(center=badge.center))
        progress = world.time / world.config.max_steps
        pygame.draw.rect(screen, _PANEL_SOFT, pygame.Rect(left, 143, 274, 7), border_radius=4)
        if progress > 0:
            pygame.draw.rect(screen, _ACCENT, pygame.Rect(left, 143, max(3, round(274 * progress)), 7),
                             border_radius=4)
        draw_text(f"{world.time}/{world.config.max_steps}", right - 70, 137, font_small, _MUTED)

        summary = (
            (f"{metrics['prey_alive']}/{len(world.prey)}", "Presas vivas", left),
            (str(metrics["predator_captures_total"]), "Capturas", left + 121),
            (str(metrics["prey_food_total"]), "Alimentos", left + 242),
        )
        for value, label, x in summary:
            draw_text(value, x, 162, font_value)
            draw_text(label, x, 190, font_tiny, _MUTED)
        draw_text(f"Supervivencia media: {fmt_number(metrics['prey_survival_mean'], 0)} pasos",
                  left, 214, font_small)
        draw_text(f"Energía media  P {fmt_number(metrics['prey_energy_mean'], 2)}  ·  D {fmt_number(metrics['predator_energy_mean'], 2)}",
                  left, 233, font_small, _MUTED)
        pygame.draw.line(screen, _LINE, (left, 257), (right, 257))

        draw_text("Políticas de decisión", left, 266, font_heading)
        for species, y, label in (("prey", 289, "Presas"), ("predator", 349, "Depredadores")):
            draw_text(label, left, y, font_small)
            for option in ("neural", "rule", "random"):
                draw_button(policy_buttons[(species, option)], _policy_label(option),
                            active=policies[species] == option,
                            disabled=option == "neural" and models[species] is None)
        draw_text("Reinicia para comparar desde el mismo inicio.", left, 405,
                  font_tiny, _MUTED)
        pygame.draw.line(screen, _LINE, (left, 427), (right, 427))

        draw_text("Agente seleccionado", left, 437, font_heading)
        if selected is None:
            draw_text("Haz clic en una presa o un depredador.", left, 471, font_small, _MUTED)
            return
        species, index = selected
        agents = world.prey if species == "prey" else world.predators
        if index >= len(agents):
            return
        agent = agents[index]
        species_name = "Presa" if species == "prey" else "Depredador"
        label = f"{species_name} {index + 1}  ·  {'Vivo' if agent.alive else 'Fuera del episodio'}"
        draw_text(label, left, 464, font_body, _PREY if species == "prey" else _PREDATOR)
        energy = font_tiny.render(f"Energía {fmt_number(agent.energy, 2)}", True, _MUTED)
        screen.blit(energy, (right - energy.get_width(), 467))
        record = last_decisions.get(selected)
        action = record["action"] if record is not None else None
        draw_text(f"Última acción: {ACTIONS[action] if action is not None else 'sin decisión aún'}",
                  left, 487, font_small, _MUTED)
        model = models[species]
        if policies[species] == "neural" and model is not None and hasattr(model, "descripcion"):
            draw_text(f"Red: {model.descripcion()}  ·  ReLU + Softmax", left, 507,
                      font_small, _ACCENT)
        else:
            draw_text(f"Política: {_policy_label(policies[species])}", left, 507,
                      font_small, _ACCENT)
        for tab, rect in tab_buttons.items():
            draw_button(rect, "Decisión" if tab == "decision" else "Entradas",
                        active=inspector_tab == tab)

        if inspector_tab == "inputs":
            observation = (record["observation"] if record is not None else
                           world.observe(species, index) if agent.alive else np.array([]))
            names = PREY_INPUTS if species == "prey" else PREDATOR_INPUTS
            draw_text(f"{len(names)} valores normalizados (capa de entrada)", left, 574,
                      font_tiny, _MUTED)
            y = 594
            for name, value in zip(names, observation):
                draw_text(_INPUT_LABELS.get(name, name.replace("_", " ")), left, y,
                          font_tiny, _MUTED)
                number = font_tiny.render(fmt_number(value, 3), True, _INK)
                screen.blit(number, (right - number.get_width(), y))
                y += 16
            return

        probabilities = record["probabilities"] if record is not None else None
        if probabilities is None:
            message = ("Ejecuta un paso para ver la decisión." if policies[species] == "neural"
                       else "Esta política no produce probabilidades.")
            draw_text(message, left, 581, font_small, _MUTED)
            if policies[species] != "neural":
                draw_text("Elige Neuronal para inspeccionar la red.", left, 606,
                          font_tiny, _MUTED)
            return
        draw_text("Salida Softmax · probabilidad por acción", left, 574, font_tiny, _MUTED)
        y = 600
        best = int(np.argmax(probabilities))
        for position, (option, probability) in enumerate(zip(ACTIONS, probabilities)):
            draw_text(option.capitalize(), left, y - 3, font_tiny,
                      _ACCENT if position == best else _INK)
            bar = pygame.Rect(left + 88, y, 205, 8)
            pygame.draw.rect(screen, _PANEL_SOFT, bar, border_radius=4)
            width = round(bar.width * float(probability))
            if width:
                pygame.draw.rect(screen, _ACCENT if position == best else _MUTED,
                                 pygame.Rect(bar.x, bar.y, width, 8), border_radius=4)
            rendered = font_tiny.render(f"{float(probability):.0%}", True, _MUTED)
            screen.blit(rendered, (right - rendered.get_width(), y - 4))
            y += 22

    def draw_help() -> None:
        dim = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        dim.fill((6, 8, 14, 190))
        screen.blit(dim, (0, 0))
        pygame.draw.rect(screen, _PAPER, help_modal, border_radius=18)
        pygame.draw.rect(screen, _ACCENT, help_modal, 2, border_radius=18)
        draw_text("Cómo usar EcosistemasNeurito", 329, 151, font_title)
        draw_button(help_close, "Cerrar")
        draw_text("Explora un episodio y observa cómo decide cada especie.", 330, 198,
                  font_body, _MUTED)
        instructions = (
            ("1", "Elige un escenario", "Abundancia, escasez o presión reinician el mundo."),
            ("2", "Elige las políticas", "Prueba red neuronal, reglas o azar por especie."),
            ("3", "Inspecciona un agente", "Haz clic en él y alterna Decisión y Entradas."),
            ("4", "Compara desde cero", "Reiniciar repite la semilla; Nueva semilla cambia el mundo."),
        )
        for number, title, detail in instructions:
            y = 235 + (int(number) - 1) * 73
            pygame.draw.circle(screen, _ACCENT, (345, y + 12), 16)
            n = font_small.render(number, True, _DARK_TEXT)
            screen.blit(n, n.get_rect(center=(345, y + 12)))
            draw_text(title, 374, y - 2, font_heading)
            draw_text(detail, 374, y + 26, font_small, _MUTED)
        pygame.draw.line(screen, _LINE, (330, 542), (950, 542))
        draw_text("Atajos", 330, 557, font_heading)
        draw_text("Espacio pausa · R reinicia · N nueva semilla · 1/2/3 escenario",
                  330, 584, font_small, _MUTED)
        draw_text("P/D cambia política · +/- velocidad · H ayuda · Esc salir",
                  330, 606, font_small, _MUTED)

    def handle_click(pos: tuple[int, int]) -> None:
        nonlocal current_scenario, paused, help_open, inspector_tab, steps_per_frame
        if help_open:
            if help_close.collidepoint(pos) or not help_modal.collidepoint(pos):
                help_open = False
            return
        if action_buttons["help"].collidepoint(pos):
            help_open = True
            return
        if action_buttons["pause"].collidepoint(pos):
            if not world.done:
                paused = not paused
            return
        if action_buttons["restart"].collidepoint(pos):
            reset_episode()
            return
        if action_buttons["new"].collidepoint(pos):
            reset_episode(new_seed=True)
            return
        for name, rect in scenario_buttons.items():
            if rect.collidepoint(pos):
                current_scenario = name
                reset_episode()
                return
        for (species, option), rect in policy_buttons.items():
            if rect.collidepoint(pos):
                set_policy(species, option)
                return
        for tab, rect in tab_buttons.items():
            if rect.collidepoint(pos):
                inspector_tab = tab
                return
        if speed_buttons["slower"].collidepoint(pos):
            steps_per_frame = max(1, steps_per_frame // 2)
            return
        if speed_buttons["faster"].collidepoint(pos):
            steps_per_frame = min(16, steps_per_frame * 2)
            return
        if world_rect.collidepoint(pos):
            choose_agent(pos)

    running = True
    while running:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if help_open:
                        help_open = False
                    else:
                        running = False
                elif event.key == pygame.K_h:
                    help_open = not help_open
                elif help_open:
                    continue
                elif event.key == pygame.K_SPACE and not world.done:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_episode()
                elif event.key == pygame.K_n:
                    reset_episode(new_seed=True)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    chosen = event.key - pygame.K_1
                    if chosen < len(scenario_names):
                        current_scenario = scenario_names[chosen]
                        reset_episode()
                elif event.key == pygame.K_p:
                    cycle_policy("prey")
                elif event.key == pygame.K_d:
                    cycle_policy("predator")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(16, steps_per_frame * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_frame = max(1, steps_per_frame // 2)

        if not paused and not help_open and not world.done:
            for _ in range(steps_per_frame):
                if world.done:
                    break
                advance()

        screen.fill(_BG)
        draw_header()
        draw_world()
        draw_sidebar()
        draw_text("Michelt Camargo · Nicoll Pérez · Sergio Ávila   |   H = atajos   ·   Esc = salir",
                  24, 760, font_small, _MUTED)
        if help_open:
            draw_help()
        pygame.display.flip()

    pygame.quit()
