# PedScrew2D: Lumbar Pedicle Screw Placement Dataset and Keypoint Detection Framework

## Introduction

In spinal fixation surgery, accurate placement of pedicle screws is crucial for ensuring patient safety and achieving optimal clinical outcomes. However, manual positioning of these screws is prone to errors, while deep learning-based screw planning is constrained by the lack of publicly available datasets and standardized evaluation frameworks. To address these issues, we developed the **PedScrew2D** dataset, containing 258 2D lumbar spine CT scans, each annotated with ten keypoints defining five essential line segments for screw planning. We then proposed a framework that simplifies the pedicle screw planning problem into a 2D keypoint detection task. Besides standard keypoint detection metrics, we introduced three specialized evaluation metrics—**AD**, **LS**, and **DS**—to provide a more detailed assessment of screw placement accuracy. Experiments demonstrated the feasibility and versatility of the proposed framework across various detection models.

## Project Structure

```
PedScrew2D/
├── data/
│	└── spine_258/# Data directory
│		├── all_data.json         # Keypoint annotation files
│   	├── annotations/          # Five-fold cross-validation annotation files
│   	├── browse_dataset/       # Dataset preview
│   	└── images/               # Image files (to be released after publication)
├── scripts/                  # Scripts directory
│   └──	{}.sh   # Five-fold cross-validation script
├── configs/                  # Configuration files
├── models/                   # Model files
├── utils/                    # Utility functions
├── README.md                 # This file
└── requirements.txt          # Dependency libraries
```

## Dataset

- **PedScrew2D** dataset consists of 258 2D lumbar spine CT scans.
- Each image is annotated with ten keypoints, defining five essential line segments for screw planning.
- Data is stored in the `data/` directory.
- You can preview the dataset using the `browser_dataset` tool.
- Image files will be made publicly available after the manuscript is published.



### Five-Fold Cross-Validation

Run the cross-validation script located in the `scripts/` directory

Ensure to modify the configuration file path and data directory according to your setup.


## Dependencies

Below are some of the key dependencies and their versions:

- **mmcv**: 2.0.0rc4
- **mmdet**: 3.3.0
- **mmengine**: 0.10.5
- **mmsegmentation**: 1.2.2
