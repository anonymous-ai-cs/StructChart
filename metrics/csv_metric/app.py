import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("csv_metric")
launch_gradio_widget(module)
