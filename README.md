# AI for Landslide Susceptibility in Himalayan & Western Ghats #

# Week 1 #

## Overview ##
This project focuses on **Landslide Susceptibility Analysis** as part of *Climate Risk and Disaster Management*.  
The Himalayan region and Western Ghats are highly vulnerable to rainfall-induced and earthquake-triggered landslides.  

By using **Python (Pandas, NumPy)**, this project performs **data loading, cleaning, and exploratory analysis** on a landslide dataset to understand the influence of various geo-environmental factors.  

##  Objectives  
- Explore landslide-related datasets.  
- Identify critical conditioning factors such as slope, rainfall, soil, and vegetation.  
- Perform **basic data exploration**:  
  - `.info()`  
  - `.describe()`  
  - `.isnull().sum()`  
- Lay the foundation for AI/ML-based landslide susceptibility modeling.

##  Dataset  
The dataset (`landslide_dataset.csv`) contains information on environmental and geological factors influencing landslides:  

- **Rainfall (mm)**  
-  **Slope Angle**  
-  **Soil Saturation**  
-  **Vegetation Cover**  
-  **Earthquake Activity**  
-  **Proximity to Water**  
-  **Soil Types (Gravel, Sand, Silt)**  
-  **Landslide (0 = No, 1 = Yes)** *(Target Variable)*  

Source:  https://www.kaggle.com/datasets/rajumavinmar/landslide-dataset?resource=download

## Tech Stack  
**Language:** Python  

**Libraries:**  
- `pandas` → Data handling  
- `numpy` → Numerical analysis  

**Environment:** VS Code (with Jupyter extension) 


##  Project Workflow  
1. **Data Collection** → Import dataset (`landslide_dataset.csv`).  
2. **Data Exploration** → Check datatypes, missing values, summary statistics.  
3. **Exploratory Data Analysis (Week 2 onward)** → Relationships between slope, rainfall, vegetation, etc.  
4. **Future AI Modeling** → Apply ML algorithms (Logistic Regression, Random Forest, XGBoost).  
5. **Outcome** → Generate Landslide Susceptibility insights for Himalayan & Western Ghats regions.  
