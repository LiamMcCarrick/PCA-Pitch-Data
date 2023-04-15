# Principal Compoennt Analysis of 2019 MLB Pitch Data: Overview
*   Utilized Principal Component Analysis to explore possible patterns and correlations in 2019 MLB pitch data
*   Dataset analyzed contained 10,000 pitches and 40 variables
*   Principal component one seems to be explained mostly by fastballs
*   Principal component two seems to be explained mostly by off speed pitches
*   65% of data captured in first two principal components
*   91% captured in first six principal components


## Code and Resources Used

**Python Version**: 3.9.7           
**Packages**: numpy, pandas, sklearn, matplotlib            
**Requirements**: `pip install -r requirements.txt`           
**Data Set**: https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018?select=2019_pitches.csv

## Features Used

*   'px', 'pz', 'start_speed', 'spin_rate', 'spin_dir'
*   'break_angle', 'break_length', 'vx0', 'vy0', 'vz0','pfx_x', 'pfx_z', 'nasty'

## Data Preprocessing

Made the following changes to clean up the data set:

## Principal Component Analysis

Steps done to create model:

## Results

Graphs from resuling PCA model:
<br>
<img src="https://github.com/LiamMcCarrick/PCA-Pitch-Data/blob/main/Variance_Explained.png" width="300" height="300">
<br>
<img src="https://github.com/LiamMcCarrick/PCA-Pitch-Data/blob/main/PCA_Biplot.png" width="300" height="300">
<br>
<img src="https://github.com/LiamMcCarrick/PCA-Pitch-Data/blob/main/PCA_Heatmap.png" width="300" height="300">
<br>