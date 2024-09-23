import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
import plotly.io as pio


def plot_sankey_diagram_from_data(data, title):
    """
    Shows a sankey diagram based on the tuple

    Args :
        - data : tuple with the following format (source, target, value, labels). 
        All these values must be arrays
    """
    source = data[0]
    target = data[1]
    value = data[2]
    labels = data[3]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(title_text=title, font_size=10)
    fig.show()


def plot_sankey_diagram_from_counter(counter: dict, title, id_to_label_fct, get_parents_fct, to_exclude: set, top_n=50):
    """
    Shows a sankey diagram based on counter object

    Args :
        - counter           : Dictionary containing the frequency of all ids (id: count)
        - title             : title of plot
        - id_to_label_fct   : Function to convert an id to a label (id are the keys of counter)
        - get_parents_fct   : Function to get the parents of an id
        - to_exclude        : Set of ids to exclude in the diagram
        - top_n             : Only consider the top-n most frequent concepts
    """
    data = counter.most_common(top_n)
    labels = []
    label_ids = []
    for item in data:
        id = item[0]
        label = id_to_label_fct(id)
        labels.append(label)
        label_ids.append(id)

    source = []
    target = []
    value = []

    for item in data:
        if item[0] in to_exclude:
            continue
        parent = get_parents_fct(item[0])
        if parent and parent[0].id in label_ids:
            parent = parent[0].id
            source.append(label_ids.index(parent))
            target.append(label_ids.index(item[0]))
            value.append(item[1])
    
    x = source, target, value, labels
    plot_sankey_diagram_from_data(x, title)
