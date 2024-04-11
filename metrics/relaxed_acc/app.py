import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("relaxed_acc")
launch_gradio_widget(module)
