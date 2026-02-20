import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter
import os
import time
import numpy as np

class RiverCrossingVisualizer:
    def __init__(self, N, k, moves):
        self.N = N
        self.k = k
        self.moves = moves
        self.left_bank = set([f'a_{i+1}' for i in range(N)] + [f'A_{i+1}' for i in range(N)])
        self.right_bank = set()
        self.boat_side = 'left'
        self.failed_move_index = None
        self.failed_people = []
        self.colors = self._generate_color_map()

    def _generate_color_map(self):
        import matplotlib.colors as mcolors
        all_people = [f'a_{i+1}' for i in range(self.N)] + [f'A_{i+1}' for i in range(self.N)]
        pastel_colors = [c for name, c in mcolors.CSS4_COLORS.items()
                         if 'light' in name.lower() or 'lavender' in name.lower() or 'misty' in name.lower()]
        np.random.seed(42)
        np.random.shuffle(pastel_colors)
        return {person: pastel_colors[i % len(pastel_colors)] for i, person in enumerate(all_people)}

    def _validate_state(self):
        for side in [self.left_bank, self.right_bank]:
            actors = {p for p in side if p.startswith('a_')}
            agents = {p for p in side if p.startswith('A_')}
            for actor in actors:
                num = actor.split('_')[1]
                if f'A_{num}' not in agents and agents:
                    self.failed_people = [actor] + list(agents)
                    return False
        return True

    def _validate_move(self, move):
        boat_set = set(move)
        if len(boat_set) == 0 or len(boat_set) > self.k:
            self.failed_people = list(boat_set)
            return False

        side = self.left_bank if self.boat_side == 'left' else self.right_bank
        if not boat_set.issubset(side):
            self.failed_people = list(boat_set - side)
            return False

        return True

    def _apply_move(self, move):
        side_from = self.left_bank if self.boat_side == 'left' else self.right_bank
        side_to = self.right_bank if self.boat_side == 'left' else self.left_bank
        for person in move:
            side_from.remove(person)
            side_to.add(person)
        self.boat_side = 'right' if self.boat_side == 'left' else 'left'

    def _draw_state(self, step, highlight=None):
        self.ax.clear()
        self.ax.set_xlim(0, 10)

        rect_height = min(0.3, 5.0 / (2 * self.N))
        y_base = 5.0
        y_max = y_base + 0.5 if self.N <= 8 else 2 * self.N * rect_height + 0.5
        self.ax.set_ylim(0, y_max)
        self.ax.axis('off')
        self.ax.set_title(f"Step {step}", fontsize=16)

        for i, person in enumerate(sorted(self.left_bank)):
            color = 'red' if highlight and person in highlight else (
                    'lightgray' if highlight else self.colors[person])
            y_pos = y_base - i * rect_height
            self.ax.add_patch(patches.Rectangle((0.5, y_pos), 2, rect_height, color=color))
            self.ax.text(1.5, y_pos + rect_height / 2, person, ha='center', va='center', fontsize=8)

        for i, person in enumerate(sorted(self.right_bank)):
            color = 'red' if highlight and person in highlight else (
                    'lightgray' if highlight else self.colors[person])
            y_pos = y_base - i * rect_height
            self.ax.add_patch(patches.Rectangle((7.5, y_pos), 2, rect_height, color=color))
            self.ax.text(8.5, y_pos + rect_height / 2, person, ha='center', va='center', fontsize=8)

        # Draw boat
        bx = 3.0 if self.boat_side == 'left' else 6.0
        self.ax.add_patch(patches.Rectangle((bx, y_base / 2), 1, 0.5, color='saddlebrown'))


    def animate(self, frame_delay=2):
        os.makedirs("videos", exist_ok=True)
        filename = input("🎥 Nombre del archivo de vídeo (sin extensión): ").strip()
        save_path = f"videos/{filename}.mp4"
        fps = 1 / frame_delay

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

        writer = FFMpegWriter(fps=1, metadata=dict(artist='RiverCrossing'), bitrate=1800)

        with writer.saving(self.fig, save_path, dpi=200):
            self._draw_state(step=0)
            writer.grab_frame()
            time.sleep(frame_delay)

            for i, move in enumerate(self.moves):
                if not self._validate_move(move):
                    print(f"\n⛔ Movimiento inválido en el paso {i + 1}. Elementos conflictivos: {self.failed_people}")
                    self._draw_state(step=i + 1, highlight=self.failed_people)
                    writer.grab_frame()
                    time.sleep(2 * frame_delay)
                    break

                self._apply_move(move)
                if not self._validate_state():
                    print(f"\n⛔ Estado inválido tras el paso {i + 1}. Conflicto con: {self.failed_people}")
                    self._draw_state(step=i + 1, highlight=self.failed_people)
                    writer.grab_frame()
                    time.sleep(2 * frame_delay)
                    break

                self._draw_state(step=i + 1)
                writer.grab_frame()
                time.sleep(frame_delay)

        print(f"\n✅ Vídeo guardado como: {save_path}")
        plt.ioff()
        plt.show()

    def is_valid_solution(self) -> bool:
        # Reiniciar el estado
        self.left_bank = set([f'a_{i+1}' for i in range(self.N)] + [f'A_{i+1}' for i in range(self.N)])
        self.right_bank = set()
        self.boat_side = 'left'
        self.failed_move_index = None
        self.failed_people = []

        for i, move in enumerate(self.moves):
            if not self._validate_move(move):
                self.failed_move_index = i
                return False

            self._apply_move(move)

            if not self._validate_state():
                self.failed_move_index = i
                return False

        # Verificar si todos han cruzado
        expected = set([f'a_{i+1}' for i in range(self.N)] + [f'A_{i+1}' for i in range(self.N)])
        return self.right_bank == expected




# # Example usage:
# # Parámetros del problema
# N = 12  # Número de actores/agentes
# k = 4  # Capacidad de la barca

# # Secuencia de movimientos de ejemplo válida para N=3 y k=2
# # Nota: esta es una secuencia simple y puede no ser óptima
# moves = [["A_1","a_1","A_2","a_2"],["A_1","a_1"],["A_3","a_3","A_4","a_4"],["A_2","a_2"],["A_5","a_5","A_6","a_6"],["A_3","a_3"],["A_7","a_7","A_8","a_8"],["A_4","a_4"],["A_9","a_9","A_10","a_10"],["A_5","a_5"],["A_11","a_11","A_12","a_12"],["A_6","a_6"],["A_1","a_1","A_2","a_2"],["A_7","a_7"],["A_3","a_3","A_4","a_4"],["A_8","a_8"],["A_5","a_5","A_6","a_6"],["A_9","a_9"],["A_7","a_7","A_8","a_8"],["A_10","a_10"],["A_9","a_9","A_10","a_10"]]

# # Crear visualizador y animar
# viz = RiverCrossingVisualizer(N, k, moves)
# viz.animate()