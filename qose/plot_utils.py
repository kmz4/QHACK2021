import matplotlib.pyplot as plt
import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout

def plot_tree(G, **kwargs):
    """

    Args:
      G: 
      **kwargs: 

    Returns:

    """
    labels = kwargs.pop('labels', True)
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(16, 8)
    pos = graphviz_layout(G, prog='dot')
    colors = nx.get_node_attributes(G, 'W')
    node_color = list(colors.values())
    vmin = min(node_color)
    vmax = max(node_color)
    nx.draw(G, pos=pos, arrows=True, with_labels=labels, cmap='OrRd', node_color=node_color, linewidths=1,
            vmin=vmin, vmax=vmax, ax=axs)
    axs.collections[0].set_edgecolor("#000000")

    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    cb = plt.colorbar(sm)
    cb.set_label('W-cost')
    axs.set_title('Tree of costs')
    plt.show()