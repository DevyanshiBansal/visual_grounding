import torch
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)
from typer import prompt

MODEL_ID = "IDEA-Research/grounding-dino-base"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CONF_THRESH = 0.35

print(f"Using device: {DEVICE}")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

model = (
    AutoModelForZeroShotObjectDetection
    .from_pretrained(MODEL_ID)
    .to(DEVICE)
)

model.eval()

def run_grounding(
        
        image,
        text_query,
        threshold=CONF_THRESH
):
    """
    image: PIL Image
    text_query: str

    Returns:
    results
    latency
    """

    start = time.time()

    inputs = processor(
        images=image,
        text=[text_query],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [image.size[::-1]]
    ).to(DEVICE)

    results = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )
    )

    latency = (
        time.time() - start
    )

    return (
        results[0],
        round(latency, 3)
    )

def draw_boxes(
        image,
        results
):
    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.imshow(image)

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["text_labels"]

    detections = []

    for box, score, label in zip(
            boxes,
            scores,
            labels):

        xmin, ymin, xmax, ymax = (
            box.cpu().numpy()
        )

        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin - 5,
            f"{label} ({score:.2f})",
            color="red",
            backgroundcolor="white"
        )

        detections.append(
            {
                "label": label,
                "confidence":
                    float(score)
            }
        )

    plt.axis("off")

    fig.canvas.draw()

    output_image = np.array(
        fig.canvas.renderer.buffer_rgba()
    )

    plt.close()

    return (
        output_image,
        detections
    )