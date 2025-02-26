import abc
import textwrap
import typing as tp
from itertools import product

import numpy as np
import outlines
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


class Color:
    BLACK = (0, 0, 0)
    GRAY = (40, 40, 40)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)


class BaseCell(abc.ABC):
    """Abstract base class for Game of Life cells."""

    def __init__(self, alive: bool = False) -> None:
        self._alive = alive

    @property
    def alive(self) -> bool:
        """Return if cell is alive."""
        return self._alive

    @alive.setter
    def alive(self, value: bool) -> None:
        """Set cell alive state."""
        self._alive = value

    def count_neighbors(self, adjacent_grid: np.ndarray) -> int:
        """Count number of alive neighbors in 3x3 grid."""
        return int(np.sum(adjacent_grid)) - int(adjacent_grid[1, 1])

    @abc.abstractmethod
    def evolve(self, adjacent_grid: np.ndarray) -> bool:
        """Calculate next state based on cell rules."""
        raise NotImplementedError


class SimpleCell(BaseCell):
    """Traditional Game of Life cell using Conway's rules."""

    def evolve(self, adjacent_grid: np.ndarray) -> bool:
        """Calculate next state based on Conway's Game of Life rules."""
        neighbor_count = self.count_neighbors(adjacent_grid)
        if self.alive:
            return 2 <= neighbor_count <= 3
        return neighbor_count == 3


class LLMCell(BaseCell):
    """Game of Life cell that uses a shared LLM to determine its next state."""

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    _model = None
    _generator = None

    def __init__(self, alive: bool = False) -> None:
        super().__init__(alive)
        if LLMCell._model is None:
            LLMCell._model = outlines.models.transformers(self.MODEL_NAME, device="cpu")
            LLMCell._generator = outlines.generate.choice(
                LLMCell._model, ["life", "death"]
            )

    def evolve(self, adjacent_grid: np.ndarray) -> bool:
        """Use shared LLM to determine next state based on visual grid."""
        ALIVE_CHAR, DEAD_CHAR = "X", "O"
        grid_repr = "\n".join(
            "|" + "|".join(ALIVE_CHAR if cell else DEAD_CHAR for cell in row) + "|"
            for row in adjacent_grid
        )

        prompt = f"""
            <|im_start|>system
            You determine the next state of a cell in Conway's Game of Life.
            <|im_end|>

            <|im_start|>user
            Current cell state: {'alive' if self._alive else 'dead'}
            3x3 grid showing cell (center) and neighbors:
            \n{grid_repr}
            What should the next state be?
            <|im_end|>
            <|im_start|>assistant
            """
        return LLMCell._generator(prompt) == "life"


class Grid:
    def __init__(
        self, rows: int, cols: int, cell_type: tp.Type[BaseCell] = SimpleCell
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.cells = np.array([[cell_type() for _ in range(cols)] for _ in range(rows)])
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
            self._history = self._history[: self._generation + 1]
            self._history.append(self.state)

    def step(self) -> None:
        """Advance grid to next generation, only processing active cells."""
        current = self.state
        cells_to_process = set()

        for i, j in self.alive_cells:
            cells_to_process.add((i, j))
            for di, dj in product((-1, 0, 1), repeat=2):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    cells_to_process.add((ni, nj))

        next_state = current.copy()
        for i, j in cells_to_process:
            adjacent_grid = np.array(
                [
                    [self._get_cell_state(current, i + di, j + dj) for dj in (-1, 0, 1)]
                    for di in (-1, 0, 1)
                ]
            )
            next_state[i, j] = self.cells[i, j].evolve(adjacent_grid)

        for i, j in cells_to_process:
            self.cells[i, j].alive = next_state[i, j]

        self._generation += 1
        self._history.append(self.state)

    def _get_cell_state(self, current: np.ndarray, i: int, j: int) -> bool:
        """Get cell state with wrapping at grid edges."""
        return current[i % self.rows, j % self.cols]

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

    def __init__(self, rows: int = 20, cols: int = 30, llm: bool = False) -> None:
        pygame.init()
        print(GAME_INSTRUCTIONS)
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Game of Life")

        self.rows = rows
        self.cols = cols
        self.cell_size = min(self.WINDOW_SIZE[0] // cols, self.WINDOW_SIZE[1] // rows)
        self.grid = Grid(rows, cols, cell_type=LLMCell if llm else SimpleCell)

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

    def update(self):
        """Update game state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                self.handle_keyboard_event(event)
            self.handle_mouse_event(event)

        if self.is_running:
            self.grid.step()

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
            text = self.font.render(stat, True, Color.RED)
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
            self.update()
            self.render()
            self.clock.tick(self.FPS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conway's Game of Life")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use a language model to determine cell state",
    )
    args = parser.parse_args()
    GameOfLife(llm=args.llm).run()
