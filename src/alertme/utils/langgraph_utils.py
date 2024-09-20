import logging

from alertme.utils.path_utils import get_output_path


def generate_graph_image(graph, filename='graph.png'):
    print("generating graph image")
    try:
        graph_image = graph.get_graph().draw_mermaid_png()
        output_path = get_output_path() / filename
        with open(output_path, 'wb') as f:
            f.write(graph_image)
        logging.info(f"Graph saved as {output_path}.")
    except Exception as e:
        logging.info(f"Failed to save graph: {e}")


def print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        logging.info(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            logging.info(msg_repr)
            _printed.add(message.id)