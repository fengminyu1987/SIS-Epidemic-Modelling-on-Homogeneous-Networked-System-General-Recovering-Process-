import networkx as nx
import matplotlib.pyplot as plt


def generate_periodic_lattice_graph(shape, size):
    """
    Generate a periodic lattice graph with the specified shape and size.

    Parameters:
    - shape (str): Shape of the lattice. Options are 'square', 'triangular', 'hexagonal'.
    - size (int): Number of nodes along one dimension (assuming a square lattice).

    Returns:
    - networkx.Graph: The generated periodic lattice graph.
    """
    G = nx.Graph()

    if shape == 'square':
        # Create nodes
        for i in range(size):
            for j in range(size):
                G.add_node(i + j * size)

        # Add edges for periodic boundary conditions
        for i in range(size):
            for j in range(size):
                # Connect with right and bottom nodes (periodic boundary)
                G.add_edge(i + j*size, i + ((j + 1) % size)*size)  # Right edge
                G.add_edge(i + j*size, (i + 1) % size + j*size)  # Bottom edge

    elif shape == 'triangular':
        # Create nodes
        for i in range(size):
            for j in range(size):
                if i % 2 == 0:
                    G.add_node((i, j, 0))
                    G.add_node((i, j, 1))
                else:
                    G.add_node((i, j, 0))
                    G.add_node((i, j, 1))

        # Add edges for periodic boundary conditions
        for i in range(size):
            for j in range(size):
                # Connect with the next neighbor
                G.add_edge((i, j, 0), ((i + 1) % size, j, 1))  # Right edge
                G.add_edge((i, j, 1), ((i - 1) % size, j, 0))  # Left edge
                G.add_edge((i, j, 0), (i, (j + 1) % size, 0))  # Bottom right edge
                G.add_edge((i, j, 1), (i, (j + 1) % size, 1))  # Upper right edge

    elif shape == 'hexagonal':
        # Create nodes
        for i in range(size):
            for j in range(size):
                G.add_node((i, j))

        # Add edges for periodic boundary conditions
        for i in range(size):
            for j in range(size):
                # Connect with right and bottom nodes (periodic boundary)
                G.add_edge((i, j), (i, (j + 1) % size))  # Right edge
                G.add_edge((i, j), ((i + 1) % size, j))  # Bottom edge
                G.add_edge((i, j), ((i - 1) % size, j))  # Top edge (for hexagonal lattice)

    else:
        raise ValueError("Invalid lattice shape. Choose from 'square', 'triangular', or 'hexagonal'.")

    return G


# # Example usage:
# shape = 'square'  # Change this to 'triangular' or 'hexagonal' for different lattice types
# size = 5  # Adjust the size of the lattice
#
# G = generate_periodic_lattice_graph(shape, size)
#
# # Draw the graph
# pos = {node: node for node in G.nodes()}  # Position nodes in a grid layout
# nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12)
# plt.title(f'{shape.capitalize()} lattice graph (size={size})')
# plt.show()

# for neiber in G.neighbors((0,0)):
#     print(neiber)
