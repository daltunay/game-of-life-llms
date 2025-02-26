import textwrap
import typing as tp
from dataclasses import dataclass
from itertools import product

import numpy as np
import pygame

GAME_INSTRUCTIONS = textwrap.dedent(
    """
    Game of Life Controls
    ====================
    Mouse:
    - Click/drag: Toggle cells (when paused)
    
    Keyboard:
    - SPACE: Start/Stop simulation
    - LEFT:  Previous generation (when paused)
    - RIGHT: Next generation (when paused)
    - ESC:   Quit game
    
    Window:
    - Close window to quit
    ====================
"""
)


class Color(tp.NamedTuple):
    BLACK: tuple[int, int, int] = (0, 0, 0)
    WHITE: tuple[int, int, int] = (255, 255, 255)
    GRAY: tuple[int, int, int] = (40, 40, 40)


@dataclass
class Cell:
    alive: bool = False

    def evolve(self, neighbor_count: int) -> bool:
        """Calculate next state based on Conway's Game of Life rules."""
        if self.alive:
            return 2 <= neighbor_count <= 3
        return neighbor_count == 3


class Grid:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.cells = np.array([[Cell() for _ in range(cols)] for _ in range(rows)])
        self._generation = 0
        self._history = [self.state]

    @property
    def state(self) -> np.ndarray:
        """Current state of the grid as boolean array."""
        return np.array([[cell.alive for cell in row] for row in self.cells])

    @property
    def generation(self) -> int:
        return self._generation

    def toggle_cell(self, row: int, col: int) -> None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.cells[row, col].alive = not self.cells[row, col].alive

    def step(self) -> None:
        """Advance grid to next generation."""
        current = self.state
        neighbor_counts = self._count_neighbors(current)

        for i, j in product(range(self.rows), range(self.cols)):
            self.cells[i, j].alive = self.cells[i, j].evolve(neighbor_counts[i, j])

        self._generation += 1
        self._history.append(self.state)

    @staticmethod
    def _count_neighbors(current: np.ndarray) -> np.ndarray:
        """Count neighbors for each cell."""
        return sum(
            np.roll(np.roll(current, dy, 0), dx, 1)
            for dx, dy in product((-1, 0, 1), repeat=2)
            if (dx, dy) != (0, 0)
        )

    def restore(self, generation: int) -> bool:
        """Restore grid to a previous generation state."""
        if not 0 <= generation < len(self._history):
            return False

        state = self._history[generation]
        for i, j in product(range(self.rows), range(self.cols)):
            self.cells[i, j].alive = state[i, j]
        self._generation = generation
        return True

    @property
    def alive_cells(self) -> list[tuple[int, int]]:
        return list(zip(*np.where(self.state)))

    @property
    def population(self) -> int:
        return sum(cell.alive for row in self.cells for cell in row)


class GameOfLife:
    WINDOW_SIZE = (800, 600)
    FONT_SIZE = 24
    FPS = 10

    def __init__(self, rows: int = 30, cols: int = 40) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Game of Life")

        self.rows = rows
        self.cols = cols
        self.cell_size = min(self.WINDOW_SIZE[0] // cols, self.WINDOW_SIZE[1] // rows)
        self.grid = Grid(rows, cols)

        self.is_running = False
        self.modified_cells = set()
        self.font = pygame.font.Font(None, self.FONT_SIZE)

    def handle_keyboard_event(self, event: pygame.event.Event) -> None:
        """Handle keyboard events."""
        if event.key == pygame.K_ESCAPE:
            pygame.quit()
            return

        if event.key == pygame.K_SPACE:
            self.is_running = not self.is_running
        elif not self.is_running:
            if event.key == pygame.K_LEFT and self.grid.generation > 0:
                self.grid.restore(self.grid.generation - 1)
            elif event.key == pygame.K_RIGHT:
                if self.grid.generation < len(self.grid._history) - 1:
                    self.grid.restore(self.grid.generation + 1)
                else:
                    self.grid.step()

    def handle_mouse_event(self, event: pygame.event.Event) -> None:
        """Handle mouse events."""
        if event.type == pygame.MOUSEBUTTONUP:
            self.modified_cells.clear()
        elif not self.is_running and (
            event.type == pygame.MOUSEBUTTONDOWN
            or (event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0])
        ):
            x, y = pygame.mouse.get_pos()
            cell_pos = (y // self.cell_size, x // self.cell_size)
            if cell_pos not in self.modified_cells:
                self.grid.toggle_cell(*cell_pos)
                self.modified_cells.add(cell_pos)

    def update(self) -> bool:
        """Update game state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                self.handle_keyboard_event(event)
            self.handle_mouse_event(event)

        if self.is_running:
            self.grid.step()
        return True

    def _draw_grid(self) -> None:
        """Draw grid lines."""
        for x in range(0, self.cols * self.cell_size + 1, self.cell_size):
            pygame.draw.line(
                self.screen, Color.GRAY, (x, 0), (x, self.rows * self.cell_size)
            )
        for y in range(0, self.rows * self.cell_size + 1, self.cell_size):
            pygame.draw.line(
                self.screen, Color.GRAY, (0, y), (self.cols * self.cell_size, y)
            )

    def _draw_cells(self) -> None:
        """Draw alive cells."""
        for i, j in self.grid.alive_cells:
            pygame.draw.rect(
                self.screen,
                Color.WHITE,
                (
                    j * self.cell_size + 1,
                    i * self.cell_size + 1,
                    self.cell_size - 1,
                    self.cell_size - 1,
                ),
            )

    def _draw_stats(self) -> None:
        """Draw game statistics."""
        stats = [
            f"Status: {'running' if self.is_running else 'paused'}",
            f"Generation: {self.grid.generation:>6}",
            f"Population: {self.grid.population:>4}",
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, Color.WHITE)
            y_pos = 10 + (i * (self.font.get_height() + 2))
            self.screen.blit(text, (10, y_pos))

    def render(self) -> None:
        """Render the game state."""
        self.screen.fill(Color.BLACK)
        self._draw_grid()
        self._draw_cells()
        self._draw_stats()
        pygame.display.flip()

    def run(self) -> None:
        """Main game loop."""
        running = True
        while running:
            if not self.update():
                break
            self.render()
            self.clock.tick(self.FPS)


if __name__ == "__main__":
    print(GAME_INSTRUCTIONS)
    GameOfLife().run()
