from PIL import Image
from grounding import (
    run_grounding,
    draw_boxes
)

image = Image.open(
    "examples/test.png"
).convert("RGB")

results, latency = run_grounding(
    image,
    "man with a cart"
)

output, detections = draw_boxes(
    image,
    results
)

print(detections)
print(latency)

import matplotlib.pyplot as plt

plt.imshow(output)
plt.axis("off")
plt.savefig("output.png", bbox_inches="tight")
plt.show()