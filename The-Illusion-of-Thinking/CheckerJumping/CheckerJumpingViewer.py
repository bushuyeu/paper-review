import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class CheckerJumpingVisualizer:
    @staticmethod
    def simulate_moves(initial_board: list, moves: list) -> list:
        """
        Simula la secuencia de movimientos en el tablero unidimensional.
        Retorna la lista de estados del tablero después de cada movimiento.
        Si hay un error, retorna los estados hasta el movimiento válido anterior.
        """
        board = initial_board.copy()
        states = [board.copy()]
        
        for i, move in enumerate(moves):
            try:
                color, from_pos, to_pos = move
                if board[from_pos] != color:
                    print(f"❌ Movimiento {i+1} inválido: {color} no está en posición {from_pos}")
                    print(f"   Estado actual: {' '.join(board)}")
                    break
                if board[to_pos] != '_':
                    print(f"❌ Movimiento {i+1} inválido: posición destino {to_pos} no está vacía")
                    print(f"   Estado actual: {' '.join(board)}")
                    break
                
                # Verificar movimiento
                if abs(from_pos - to_pos) == 1:
                    # Slide
                    pass
                elif abs(from_pos - to_pos) == 2:
                    # Jump
                    mid_pos = (from_pos + to_pos) // 2
                    opposite = 'B' if color == 'R' else 'R'
                    if board[mid_pos] != opposite:
                        print(f"❌ Movimiento {i+1} inválido: salto inválido, no hay {opposite} en posición {mid_pos}")
                        print(f"   Estado actual: {' '.join(board)}")
                        break
                else:
                    print(f"❌ Movimiento {i+1} inválido: distancia {abs(from_pos - to_pos)} no permitida")
                    print(f"   Estado actual: {' '.join(board)}")
                    break
                
                # Ejecutar
                board[from_pos] = '_'
                board[to_pos] = color
                states.append(board.copy())
                
            except Exception as e:
                print(f"❌ Error en movimiento {i+1}: {e}")
                print(f"   Estado actual: {' '.join(board)}")
                break
        
        return states

    @staticmethod
    def animate(initial_board: list, moves: list):
        """
        Crea una animación del tablero evolucionando con los movimientos.
        Muestra todos los estados válidos hasta donde se pueda simular.
        """
        states = CheckerJumpingVisualizer.simulate_moves(initial_board, moves)
        
        if len(states) <= 1:
            print("⚠️ No hay movimientos válidos para animar.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(-0.5, len(initial_board) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Colores
        color_map = {'R': 'red', '_': 'white', 'B': 'blue'}
        
        def update(frame):
            ax.clear()
            ax.set_xlim(-0.5, len(initial_board) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            board = states[frame]
            for i, piece in enumerate(board):
                color = color_map.get(piece, 'black')
                ax.add_patch(plt.Rectangle((i-0.4, -0.4), 0.8, 0.8, color=color, ec='black'))
                if piece != '_':
                    ax.text(i, 0, piece, ha='center', va='center', fontsize=20, color='white')
            
            title = f"Paso {frame}: {' '.join(board)}"
            if frame == len(states) - 1 and len(states) - 1 < len(moves):
                title += " (ERROR DETECTADO)"
            ax.set_title(title)
        
        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=1000, repeat=False)
        plt.show()

    @staticmethod
    def print_board(board: list):
        """
        Imprime el tablero en texto.
        """
        print(' '.join(board))

# Ejemplo de uso
if __name__ == "__main__":
    N = 7
    initial_board = ['R'] * N + ['_'] + ['B'] * N
    # Ejemplo de movimientos (ajusta según necesidad)
    moves = [['R', 6, 7], ['B', 8, 6], ['B', 9, 8], ['R', 7, 9], ['R', 5, 7], ['R', 4, 5], ['B', 6, 4], ['B', 8, 6], ['B', 10, 8], ['B', 11, 10], ['R', 9, 11], ['R', 7, 9], ['R', 5, 7], ['R', 3, 5], ['R', 2, 3], ['B', 4, 2], ['B', 6, 4], ['B', 8, 6], ['B', 10, 8], ['B', 12, 10], ['B', 13, 12], ['R', 11, 13], ['R', 9, 11], ['R', 7, 9], ['R', 5, 7], ['R', 3, 5], ['R', 1, 3], ['R', 0, 1], ['B', 2, 0], ['B', 4, 2], ['B', 6, 4], ['B', 8, 6], ['B', 10, 8], ['B', 12, 10], ['B', 14, 12], ['R', 13, 14], ['R', 11, 13], ['R', 9, 11], ['R', 7, 9], ['R', 5, 7], ['R', 3, 5], ['R', 1, 3], ['B', 2, 1], ['B', 4, 2], ['B', 6, 4], ['B', 8, 6], ['B', 10, 8], ['B', 12, 10], ['R', 11, 12], ['R', 9, 11], ['R', 7, 9], ['R', 5, 7], ['R', 3, 5], ['B', 4, 3], ['B', 6, 4], ['B', 8, 6], ['B', 10, 8], ['R', 9, 10], ['R', 7, 9], ['R', 5, 7], ['B', 6, 5], ['B', 8, 6], ['R', 7, 8]]
    
    print("Tablero inicial:")
    CheckerJumpingVisualizer.print_board(initial_board)
    
    # Simular movimientos y mostrar estados hasta donde sea posible
    states = CheckerJumpingVisualizer.simulate_moves(initial_board, moves)
    print(f"\nEstados simulados ({len(states)} de {len(moves) + 1} posibles):")
    for i, state in enumerate(states):
        print(f"Paso {i}: {' '.join(state)}")
    
    # Animar siempre, incluso si hay errores
    print("\n🎥 Mostrando animación de estados válidos...")
    CheckerJumpingVisualizer.animate(initial_board, moves)
