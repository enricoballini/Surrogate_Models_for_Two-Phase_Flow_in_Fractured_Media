"""Draw a field on the mesh of a scenario."""

from matplotlib.collections import LineCollection

from src.data.dataset import N_FRACTURE_CELLS, N_MATRIX_CELLS, load_mesh


def draw_on_mesh(axes, field, split, scenario, **color_options):
    """Draw one value per cell and return the image, needed to add a colorbar.

    The matrix cells are triangles and the fracture cells are segments drawn on top of
    them. The intersections of the fractures are points and are not drawn.
    """
    matrix_nodes, matrix_triangles, fracture_segments, _ = load_mesh(split)
    nodes = matrix_nodes[scenario]
    image = axes.tripcolor(
        nodes[:, 0],
        nodes[:, 1],
        matrix_triangles,
        facecolors=field[:N_MATRIX_CELLS],
        **color_options,
    )

    fractures = LineCollection(fracture_segments[scenario], linewidths=2)
    fractures.set_array(field[N_MATRIX_CELLS : N_MATRIX_CELLS + N_FRACTURE_CELLS])
    fractures.set_cmap(color_options["cmap"])
    fractures.set_clim(color_options["vmin"], color_options["vmax"])
    axes.add_collection(fractures)

    axes.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    axes.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    return image
