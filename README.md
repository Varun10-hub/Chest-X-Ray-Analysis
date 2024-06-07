# Chest X-ray Analysis

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. See the [LICENSE.md](LICENSE.md) file for details.

## Repository Structure

- **notebooks**: Contains Jupyter notebooks with code and analysis.
- **assets**: Contains visualization results for all models across both datasets.
- **LICENSE.md**: License information.
- **README.md**: Project overview and information.
- **requirements.txt**: Dependencies and packages required for the project.

## Table of Contents

- [Introduction](#introduction)
- [Project Goals](#project-goals)
- [Models Used](#models-used)
- [Performance Evaluation](#performance-evaluation)
- [Datasets](#datasets)
- [Reference Paper](#reference-paper)
- [Acknowledgments](#acknowledgments)

## Introduction

The project investigates the performance of machine learning models for the analysis of chest X-ray images. The focus is on comparing the performance of classification and segmentation models on two separate datasets:

- **Full Chest X-rays**: This dataset contains chest X-ray images of the entire thorax.
- **Segmented Lung X-rays**: This dataset contains chest X-ray images where only the lungs have been segmented and isolated.

## Project Goals

The project aimed to:

- Evaluate the performance of different deep learning models for classifying chest X-ray images (COVID-19, Pneumonia, or Healthy/Normal). This evaluation involved training the models with default weights on the first dataset, performing inference and interpretability analysis, and computing raw performance metrics. The same models were then trained with default weights on the second dataset and evaluated similarly.
- Compare the performance of classification models for each dataset.
- Analyze the interpretability of the classification models using techniques like Grad-CAM and Saliency Maps.
- Develop a custom approach for interpreting the segmentation model's predictions. (Note: This exploration is supplementary and focuses on the YOLO model trained on the second dataset, but used for inference on both datasets.)

## Models Used

The project employed the following deep learning models:

### Classification Models

- ResNet18
- ResNet34
- DenseNet161
- InceptionV3

### Segmentation Model

- YOLOv8n-seg by Ultralytics (for supplementary exploration)

## Performance Evaluation

The models were evaluated on various metrics including accuracy, precision, recall, F1-score, and interpretability. A detailed analysis of the results, comparisons, and visualizations can be found in the accompanying research paper.

## Datasets

The project utilizes two external chest X-ray datasets. Due to size constraints, the datasets are not included in this repository. Information regarding the data sources, is provided in the research paper.

## Reference Paper

This project is a solo effort documented in a comprehensive research paper that details the methodology, findings, results, and visualizations. You can find the paper on [Google Drive](https://drive.google.com/file/d/1YKOquZfB0PWgLXoil0_miafx5ck5v2lf/view?usp=sharing).

## Acknowledgments

I would like to express my sincere gratitude to my mentor, Ivan Felipe Rodriguez, for their invaluable guidance, support, and encouragement throughout this project.