import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

class BlocksWorldViewer:
    def __init__(self, initial_state, target_state=None):
        """
        Inicializa el visor de BlocksWorld.
        
        Args:
            initial_state: Lista de listas representando los stacks iniciales
            target_state: Lista de listas representando el estado objetivo (opcional)
        """
        self.initial_state = [stack.copy() for stack in initial_state]
        self.target_state = [stack.copy() for stack in target_state] if target_state else None
        self.states = [self.initial_state]
        self.current_state = [stack.copy() for stack in initial_state]
        self.num_stacks = len(initial_state)
        
        # Colores para los bloques (letras A-Z)
        self.block_colors = {}
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            self.block_colors[char] = colors[i % 20]
    
    def simulate_moves(self, moves, animate=True):
        """
        Simula una secuencia de movimientos y automáticamente los anima.
        
        Args:
            moves: Lista de movimientos en formato [block, from_stack, to_stack]
            animate: Si True, muestra animación paso a paso (por defecto True)
            
        Returns:
            bool: True si todos los movimientos fueron válidos y se alcanzó el objetivo, False si hubo error
        """
        self.states = [[stack.copy() for stack in self.initial_state]]
        self.current_state = [stack.copy() for stack in self.initial_state]
        
        print(f"🎯 Iniciando simulación con {len(moves)} movimientos...")
        print(f"Estado inicial: {self.initial_state}")
        if self.target_state:
            print(f"Estado objetivo: {self.target_state}")
        print("-" * 50)
        
        for i, move in enumerate(moves):
            block, from_stack, to_stack = move
            
            # Verificar validez del movimiento
            try:
                # Verificar que el stack origen no esté vacío
                if not self.current_state[from_stack]:
                    print(f"❌ Error en movimiento {i+1}: Stack {from_stack} está vacío")
                    break
                
                # Verificar que el bloque esté en la cima
                if self.current_state[from_stack][-1] != block:
                    expected = self.current_state[from_stack][-1]
                    print(f"❌ Error en movimiento {i+1}: Se esperaba '{expected}' en la cima del stack {from_stack}, pero se intentó mover '{block}'")
                    break
                
                # Verificar índices válidos
                if from_stack < 0 or from_stack >= self.num_stacks or to_stack < 0 or to_stack >= self.num_stacks:
                    print(f"❌ Error en movimiento {i+1}: Índices de stack inválidos ({from_stack} -> {to_stack})")
                    break
                
                # Ejecutar movimiento
                moved_block = self.current_state[from_stack].pop()
                self.current_state[to_stack].append(moved_block)
                
                # Guardar estado
                self.states.append([stack.copy() for stack in self.current_state])
                
                print(f"✅ Movimiento {i+1}: '{block}' desde Stack {from_stack} → Stack {to_stack}")
                print(f"   Estado actual: {self.current_state}")
                
            except Exception as e:
                print(f"❌ Error inesperado en movimiento {i+1}: {e}")
                break
        
        # Verificar si se alcanzó el objetivo
        all_moves_valid = len(self.states) == len(moves) + 1
        goal_reached = self.target_state and self.current_state == self.target_state
        
        print("-" * 50)
        print(f"📊 Resumen:")
        print(f"   • Movimientos válidos: {len(self.states) - 1}/{len(moves)}")
        print(f"   • Estado final: {self.current_state}")
        
        if self.target_state:
            if goal_reached:
                print(f"   • ✅ ¡Objetivo alcanzado!")
            else:
                print(f"   • ❌ Objetivo no alcanzado")
                print(f"   • Estado esperado: {self.target_state}")
        
        if animate and len(self.states) > 1:
            print(f"🎬 Iniciando animación de {len(self.states)} estados...")
            self.animate()
        
        # Retornar True solo si todos los movimientos fueron válidos Y se alcanzó el objetivo
        return all_moves_valid and goal_reached
    
    def draw_state(self, state, ax, title="BlocksWorld State"):
        """
        Dibuja un estado específico del puzzle.
        
        Args:
            state: Estado a dibujar
            ax: Axes de matplotlib
            title: Título del gráfico
        """
        ax.clear()
        ax.set_xlim(-0.5, self.num_stacks - 0.5)
        
        # Encontrar la altura máxima
        max_height = max(len(stack) for stack in state) if any(state) else 1
        ax.set_ylim(-0.1, max_height + 0.5)
        
        # Dibujar stacks
        for stack_idx, stack in enumerate(state):
            # Dibujar base del stack
            base = patches.Rectangle((stack_idx - 0.4, -0.1), 0.8, 0.1, 
                                   facecolor='gray', edgecolor='black')
            ax.add_patch(base)
            
            # Dibujar bloques
            for block_idx, block in enumerate(stack):
                color = self.block_colors.get(block, 'lightgray')
                
                # Bloque
                rect = patches.Rectangle((stack_idx - 0.3, block_idx), 0.6, 0.8,
                                       facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                # Etiqueta del bloque
                ax.text(stack_idx, block_idx + 0.4, block, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Etiquetas de stacks
        for i in range(self.num_stacks):
            ax.text(i, -0.3, f'Stack {i}', ha='center', va='top', fontsize=10)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    def show_states_comparison(self):
        """
        Muestra una comparación entre estado inicial, final y objetivo.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Estado inicial
        self.draw_state(self.initial_state, axes[0], "Estado Inicial")
        
        # Estado final
        final_state = self.states[-1] if self.states else self.initial_state
        self.draw_state(final_state, axes[1], "Estado Final")
        
        # Estado objetivo (si existe)
        if self.target_state:
            self.draw_state(self.target_state, axes[2], "Estado Objetivo")
            
            # Verificar si se alcanzó el objetivo
            if final_state == self.target_state:
                axes[1].set_title("Estado Final ✅", fontsize=16, fontweight='bold', color='green')
            else:
                axes[1].set_title("Estado Final ❌", fontsize=16, fontweight='bold', color='red')
        else:
            axes[2].text(0.5, 0.5, 'No hay\nestado objetivo\ndefinido', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=14)
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def animate(self, interval=2000):
        """
        Crea una animación de todos los estados.
        
        Args:
            interval: Tiempo entre frames en milisegundos
        """
        if len(self.states) <= 1:
            print("No hay suficientes estados para animar")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update(frame):
            state = self.states[frame]
            title = f"Paso {frame}"
            if frame == 0:
                title += " - Estado Inicial"
            elif frame == len(self.states) - 1:
                title += " - Estado Final"
                if self.target_state and state == self.target_state:
                    title += " ✅ ¡Objetivo Alcanzado!"
                elif self.target_state:
                    title += " ❌ Objetivo No Alcanzado"
            else:
                title += f" - Después del movimiento {frame}"
            
            self.draw_state(state, ax, title)
            
            # Agregar información del movimiento actual en la parte inferior
            if frame > 0 and frame <= len(self.states) - 1:
                # Buscar qué movimiento se realizó comparando estados
                prev_state = self.states[frame - 1]
                current_state = self.states[frame]
                
                # Encontrar diferencias entre estados
                move_info = self._find_move_between_states(prev_state, current_state)
                if move_info:
                    ax.text(0.5, -0.15, f"Movimiento: {move_info}", 
                           ha='center', va='top', transform=ax.transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor="lightblue", alpha=0.8))
            
            return []
        
        anim = FuncAnimation(fig, update, frames=len(self.states), 
                           interval=interval, blit=False, repeat=True)
        
        plt.show()
        return anim
    
    def _find_move_between_states(self, prev_state, current_state):
        """
        Encuentra qué movimiento se realizó entre dos estados consecutivos.
        
        Args:
            prev_state: Estado anterior
            current_state: Estado actual
            
        Returns:
            str: Descripción del movimiento o None si no se encuentra
        """
        for stack_idx in range(len(prev_state)):
            # Buscar stack que perdió un bloque
            if len(prev_state[stack_idx]) > len(current_state[stack_idx]):
                from_stack = stack_idx
                moved_block = prev_state[stack_idx][-1]  # Último bloque del stack anterior
                
                # Buscar stack que ganó el bloque
                for dest_idx in range(len(current_state)):
                    if (len(current_state[dest_idx]) > len(prev_state[dest_idx]) and
                        current_state[dest_idx][-1] == moved_block):
                        to_stack = dest_idx
                        return f"Mover '{moved_block}' del Stack {from_stack} al Stack {to_stack}"
        
        return None
    
    def show_current_state(self):
        """
        Muestra solo el estado actual.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        self.draw_state(self.current_state, ax, "Estado Actual")
        plt.show()

# Ejemplo de uso y testing
if __name__ == "__main__":
    # Configuración del puzzle (N=4)
    initial_state = [['A', 'B'], ['C', 'D'], []]
    target_state = [[], [], ['D', 'B', 'C', 'A']]
    
    print("🔧 Configuración del BlocksWorld:")
    print(f"   Estado inicial: {initial_state}")
    print(f"   Estado objetivo: {target_state}")
    print("=" * 60)
    
    # Crear el viewer
    viewer = BlocksWorldViewer(initial_state, target_state)
    
    # ===== SECCIÓN PARA COPIAR/PEGAR MOVIMIENTOS =====
    # Copia aquí los movimientos que quieras probar:
    # Formato: [["bloque", stack_origen, stack_destino], ...]
    
    # Ejemplo 1: Solución correcta
    moves = [['D', 1, 2], ['B', 0, 2], ['C', 1, 2]]
    
    # Ejemplo 2: Solución con error (descomenta para probar)
    # moves = [['A', 0, 2], ['B', 0, 2], ['C', 1, 2], ['D', 1, 2]]  # Error: A no está en la cima
    
    # Ejemplo 3: Movimientos parciales (descomenta para probar)
    # moves = [['D', 1, 2], ['B', 0, 2]]  # Solo los primeros 2 movimientos
    
    # ==================================================
    
    print(f"🚀 Probando secuencia de {len(moves)} movimientos:")
    for i, move in enumerate(moves):
        print(f"   {i+1}. Mover '{move[0]}' del Stack {move[1]} al Stack {move[2]}")
    print()
    
    # Simular y animar
    success = viewer.simulate_moves(moves)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ¡ÉXITO! Todos los movimientos fueron válidos y se alcanzó el objetivo.")
    else:
        print("⚠️  La simulación no completó todos los movimientos o no se alcanzó el objetivo.")
    
    print("\n💡 Para probar otros movimientos:")
    print("   1. Edita la variable 'moves' en este archivo")
    print("   2. Ejecuta: python3 BlocksWorldViewer.py")
    print("   3. La animación se mostrará automáticamente")
