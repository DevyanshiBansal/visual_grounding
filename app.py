import gradio as gr
import numpy as np
from PIL import Image

from grounding import (
    run_grounding,
    draw_boxes,
    DEVICE
)


def predict(image, query, threshold):

    # Gradio gives numpy image
    image = Image.fromarray(image)

    # Run model
    results, latency = run_grounding(
        image,
        query,
        threshold
    )

    # Draw boxes
    output_image, detections = draw_boxes(
        image,
        results
    )

    ##################################################
    # ANALYTICS
    ##################################################

    num_objects = len(detections)

    scores = [
        d["confidence"]
        for d in detections
    ]

    avg_conf = (
        round(
            sum(scores) / len(scores),
            3
        )
        if scores
        else 0
    )

    max_conf = (
        round(
            max(scores),
            3
        )
        if scores
        else 0
    )

    min_conf = (
        round(
            min(scores),
            3
        )
        if scores
        else 0
    )

    variance = (
        round(
            np.var(scores),
            4
        )
        if scores
        else 0
    )

    ##################################################
    # ANALYTICS STRING
    ##################################################

    analytics = f"""
MODEL INFORMATION
────────────────────────────
Model              : GroundingDINO Base
Task               : Zero-Shot Visual Grounding
Image Encoder      : Swin Transformer
Text Encoder       : CLIP
Framework          : HuggingFace Transformers
Device             : {DEVICE}

INFERENCE STATISTICS
────────────────────────────
Prompt             : {query}
Inference Time     : {latency:.2f} sec
Threshold Used     : {threshold}
Objects Detected   : {num_objects}

CONFIDENCE ANALYTICS
────────────────────────────
Average Confidence : {avg_conf}
Maximum Confidence : {max_conf}
Minimum Confidence : {min_conf}
Variance           : {variance}
"""

    return (
        output_image,
        str(detections),
        analytics
    )


##################################################
# UI
##################################################

with gr.Blocks() as demo:

    gr.Markdown("# VisionGround")

    gr.Markdown(
        "Zero-Shot Visual Grounding using Vision-Language Models"
    )

    with gr.Row():

        image_input = gr.Image(
            type="numpy",
            label="Image"
        )

        with gr.Column():

            query = gr.Textbox(
                label="Text Query",
                placeholder="person with shopping cart"
            )

            threshold = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.35,
                step=0.01,
                label="Confidence Threshold"
            )

            btn = gr.Button(
                "Ground Objects"
            )

    output_image = gr.Image(
        label="Annotated Image"
    )

    detections = gr.Textbox(
        label="Detections",
        lines=5
    )

    analytics = gr.Textbox(
        label="Model Analytics",
        lines=20
    )

    btn.click(
        fn=predict,
        inputs=[
            image_input,
            query,
            threshold
        ],
        outputs=[
            output_image,
            detections,
            analytics
        ]
    )

demo.launch()