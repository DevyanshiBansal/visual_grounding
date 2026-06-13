import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ======================================
# Import grounding.py
# ======================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(BASE_DIR)

from grounding import run_grounding

# ======================================
# Load Image
# ======================================

image_path = os.path.join(
    BASE_DIR,
    "examples",
    "test.png"
)

image = Image.open(
    image_path
).convert("RGB")

# ======================================
# Prompt Set
# ======================================

prompts = [

    # Baseline
    "phone",
    "mobile phone",
    "smartphone",
    "phones",

    # Attribute prompts
    "black mobile phone",
    "black smartphone",
    "phone showing launch soon",

    # Long prompts
    "a black mobile phone lying on the table",
    "a smartphone displaying the text launch soon",
    "a mobile phone placed on a wooden table",
    "a mobile phone showing an advertisement",

    # Scene prompts
    "multiple mobile phones on a table",
    "group of mobile phones",
    "phones surrounding a launch soon poster",

    # Very long prompts
    "a black smartphone placed on a table and displaying the words launch soon on its screen",

    "a group of black mobile phones placed around a poster containing the text launch soon",

    "multiple smartphones with illuminated screens lying around a rectangular poster"
]

# ======================================
# Thresholds
# ======================================

thresholds = np.arange(
    0.1,
    1.0,
    0.1
)

# ======================================
# Run Experiments
# ======================================

results = []

for prompt in prompts:

    print("\n")
    print("=" * 70)
    print("Prompt:", prompt)

    for threshold in thresholds:

        print(
            f"Threshold: {threshold:.1f}"
        )

        output, latency = run_grounding(
            image,
            prompt,
            threshold=round(
                threshold,
                1
            )
        )

        scores = [
            float(s)
            for s in output["scores"]
        ]

        detections = len(scores)

        if detections > 0:

            avg_conf = round(
                np.mean(scores),
                3
            )

            max_conf = round(
                np.max(scores),
                3
            )

            min_conf = round(
                np.min(scores),
                3
            )

            variance = round(
                np.var(scores),
                3
            )

        else:

            avg_conf = 0
            max_conf = 0
            min_conf = 0
            variance = 0

        results.append(
            {
                "Prompt": prompt,
                "Threshold": round(
                    threshold,
                    1
                ),
                "Detections": detections,
                "Average Confidence": avg_conf,
                "Maximum Confidence": max_conf,
                "Minimum Confidence": min_conf,
                "Variance": variance,
                "Inference Time": latency
            }
        )

# ======================================
# Dataframe
# ======================================

df = pd.DataFrame(results)

csv_path = os.path.join(
    BASE_DIR,
    "sensitivity_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\nCSV Saved:")
print(csv_path)

# ======================================
# Question 1
# Highest Confidence
# At Highest Threshold
# ======================================

working = df[
    df["Detections"] > 0
]

if len(working) > 0:

    highest_threshold = (
        working["Threshold"].max()
    )

    highest_df = working[
        working["Threshold"]
        ==
        highest_threshold
    ]

    best_high_prompt = highest_df.loc[
        highest_df[
            "Average Confidence"
        ].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print("BEST PROMPT AT HIGHEST THRESHOLD")
    print("=" * 70)

    print(
        "Prompt:",
        best_high_prompt["Prompt"]
    )

    print(
        "Threshold:",
        best_high_prompt["Threshold"]
    )

    print(
        "Confidence:",
        best_high_prompt[
            "Average Confidence"
        ]
    )

# ======================================
# Question 2
# Best Working Prompts
# ======================================

prompt_summary = df.groupby(
    "Prompt"
).agg(
    {
        "Average Confidence": "mean",
        "Detections": "mean",
        "Threshold": "max"
    }
)

prompt_summary["Score"] = (

        prompt_summary[
            "Average Confidence"
        ]

        *

        prompt_summary[
            "Detections"
        ]

        *

        prompt_summary[
            "Threshold"
        ]
)

prompt_summary = (
    prompt_summary
    .sort_values(
        "Score",
        ascending=False
    )
)

print("\n")
print("=" * 70)
print("TOP 5 PROMPTS")
print("=" * 70)

print(
    prompt_summary.head(5)
)

# ======================================
# Question 3
# Worst Prompts
# ======================================

print("\n")
print("=" * 70)
print("WORST 5 PROMPTS")
print("=" * 70)

print(
    prompt_summary.tail(5)
)

# ======================================
# Question 4
# Failure Threshold
# ======================================

threshold_summary = (
    df.groupby(
        "Threshold"
    )["Detections"]
    .sum()
)

failure_threshold = None

for t, d in threshold_summary.items():

    if d == 0:

        failure_threshold = t
        break

print("\n")
print("=" * 70)
print("FAILURE THRESHOLD")
print("=" * 70)

if failure_threshold:

    print(
        f"No prompt works above threshold {failure_threshold}"
    )

else:

    print(
        "At least one prompt survives all thresholds."
    )

# ======================================
# Heatmap
# ======================================

pivot = df.pivot_table(
    index="Prompt",
    columns="Threshold",
    values="Average Confidence"
)

plt.figure(
    figsize=(14, 10)
)

plt.imshow(
    pivot,
    aspect="auto"
)

plt.colorbar(
    label="Average Confidence"
)

plt.xticks(
    range(
        len(pivot.columns)
    ),
    pivot.columns
)

plt.yticks(
    range(
        len(pivot.index)
    ),
    pivot.index
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Prompt"
)

plt.title(
    "Prompt vs Threshold vs Confidence"
)

plt.tight_layout()

plt.show()

# ======================================
# Threshold Sensitivity Plot
# ======================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    threshold_summary.index,
    threshold_summary.values,
    marker="o"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Total Detections"
)

plt.title(
    "Threshold Sensitivity Analysis"
)

plt.grid()

plt.show()