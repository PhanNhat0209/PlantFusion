# Usage Guide

## 1. Download Dataset
- Open the `data.txt` file to get the dataset download link.
- Download the dataset and place it in the same directory as the code files.

## 2. Update Dataset Path
- Open the `train.py` file and update the dataset path as follows:
  ```python
  data_dir = "/kaggle/input/plant-augment/data"
  output_dir = "/kaggle/working/"
  ```

## 3. Run Training Process
- Run the following command to start training:
  ```bash
  python train.py
  ```