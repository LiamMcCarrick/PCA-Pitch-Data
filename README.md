# Principal Compoennt Analysis of 2019 MLB Pitch Data: Overview
*   Utilized Principal Component Analysis to explore possible patterns and associations in 2019 MLB pitch data
*   Dataset analyzed contained 10,000 pitches and 40 variables
*   Break length, velocity towards homeplate and vertical velocity have significant influence on offspeed pitches
*   Horizontal movement, spin rate and start speed have significant influence on fastballs
*   65% of data captured in first two principal components
*   91% captured in first six principal components


## Code and Resources Used

**Python Version**: 3.9.7           
**Packages**: numpy, pandas, sklearn, matplotlib, seaborn            
**Requirements**: `pip install -r requirements.txt`           
**Data Set**: https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018?select=2019_pitches.csv           
**Biplot/Heatmap Information**:         
https://ostwalprasad.github.io/machine-learning/PCA-using-python.html           
https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e           
**Pitch Type Definitions**: https://baseballsavant.mlb.com/csv-docs#des         
**Textbook**: Python Data Science Handbook by Jacob T. Vanderplas

## Features Used

The definitions of each variable can be viewed in the Pitch Type Definition link in the resource section above.
-   The initial dataset contained 40 variables for each pitch recorded
-   I removed variables that were not an indicator of the type of pitch thrown          

That left me with the following 14 variables:                      
*   'px', 'pz', 'start_speed', 'spin_rate', 'spin_dir', 'pitch_type'
*   'break_angle', 'break_length', 'vx0', 'vy0', 'vz0','pfx_x', 'pfx_z', 'nasty'

## Data Preprocessing

Before running the PCA transform, I needed to clean up the data. Listed below are the changes made to the dataset:          
*   Dropped rows that contained NANs
*   Recoded the 15 pitch types to just fastballs, offspeed and not applicable pitches
*   Removed all rows that had NA pitch types
*   The pitch data was standardized as various variables had different scales

## Principal Component Analysis

*   Performed PCA to reveal any possible contrasts and assocations within the data
*   Reduced the dimension of the data to bring out the important variables 

The following graph is the explained variance of each principal component.        

<img src="https://github.com/LiamMcCarrick/PCA-Pitch-Data/blob/main/Variance_Explained.png" width="450" height="400">

*   The explained variance is the amount of variance in the dataset that is contained within each principal component
*   65% of data captured in first two principal components
*   91% captured in first six principal components

The following graph is a heatmap between the principal components and pitch types.

<img src="https://github.com/LiamMcCarrick/PCA-Pitch-Data/blob/main/PCA_Heatmap.png" width="450" height="400">

*   The heatmap shows which variables influence each principal component the most
*   Break length, vertical velocity (vy0), velocity towards homeplate (vz0) and horizontal movement (pfx_x) influence the first principal component
*   Horizontal movement (pfx_x), spin rate, start speed, and vertical movement (pfx_z) inflence the second principal component

The following graph is a biplot of the principal components.

<img src="https://github.com/LiamMcCarrick/PCA-Pitch-Data/blob/main/PCA_Biplot.png" width="450" height="400">

*   Green = Fastballs | Red = Offspeed pitches
*   A biplot is a helpful all in one visualization of principal component analysis
*   The direction of the arrows show the influence and weight of the features in the principal components
*   The angle of the lines indicate how much each variable correlates with one another in each principal component
*   Break length, velocity towards homeplate and vertical velocity have significant influence on offspeed pitches
*   Horizontal movement, spin rate and start speed have significant influence on fastballs