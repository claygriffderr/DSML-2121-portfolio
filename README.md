# Data Analysis Portfolio Dashboard
**Author**: Clay Griffin-Derr
**contact**: cgriff60@msudenver.edu
**LinkedIn**: Clay Griffin-Derr [www.linkedin.com/in/clay-griffin-derr-3542952b8]
**Date**: 11-18-26

## Description:
This is an interactive Streamlit app written for DSML 2120 - Data Visualization (11/18/2026). It allows users to get to know the author, and explore the Red Wine Kaggle Dataset using interactive charts. 

### Pages:
- Bio: contains information about the author
- EDA: Interactive charts for understanding data distributions and variable correlations
- Dashboard: Interactive charts for exploring simple prediction methods on the dataset
- Future Work: Brief explanation of next steps and author's process

### Dataset:
Kaggle Red Wine Dataset [https://www.kaggle.com/datasets/lovishbansal123/red-wine-quality]

This dataset contains 1600 wines with numeric variables for attributes like pH, alcohol content and more. The target variable is a quality metric (0-10), assigned by wine professionals. 

**Limitations**: Data has been de-identified, with no geographic, brand or other categorical information. 
**Preprocessing**: The dataset contains 0 missing values. Several variables contain signifigant outliers, so the dashboard allows users to exclude these values for analysis. 

### Requirements:
Consult the requirements.txt file for information on required packages to run this application. 

### AI Assistance Acknowledgement:
Claude Opus 5.5 on medium and high settings was used to generate code for dashboard visuals. Prompts used are included in comments at the top of the scripts for the EDA and dashboard pages. Structure and content were specified by the user, and resulting code was tested and refined manually. All insights, biography and "how to read" sections were written by the author only. 
