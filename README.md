# Policing Equity Analysis

This repository might not run on a Windows machine. It should work fine on Ubuntu. You can use WSL.

## Objective of this repository:
This repository analysis the dataset "Data Science for Good: Center for Policing Equity"
available at: [Kaggle - Data Science for Good](https://www.kaggle.com/datasets/center-for-policing-equity/data-science-for-good)

## Supporting Documents
The following files contain additional information: \
### [DATA_STRUCTURE](DATA_STRUCTURE.md)
Contains information about the structure of all files in from [Kaggle - Data Science for Good](https://www.kaggle.com/datasets/center-for-policing-equity/data-science-for-good) and a brief analysis.

### [POLICING_REPORT_ANALYSIS](POLICING_REPORT_ANALYSIS.md)
Contains a detailed description of the steps conducted and the respective results.
The information here are to support the description in the paper.

### [Univariate Statistics](stats)
This directory contains several plots and detailed statistical information about the features after pre-processing.

### [Clustering Results](clustering/cluster_stats)
This directory contains the results of the clustering of groups. It comes along with different visualizations.

### [final_values] (final_values.csv)
This file contains the clusters of the groups. The dataframe is reduced to the cluster results of the groups and complete clusters.

### [first_clustering] (first_clustering.csv)
This file contains the features and the cluster results of the group clustering.

## Main
The main.py file can be used to decide which analysis to run.

## Structure
### 1. Metadata Analysis
Run: metadata_analysis.py

Objective: Check for the ACS reports - all types and all departments.

Result:
- Each type of report uses the same codes and descriptions across all departments.
- Each department contains the same types of reports.
- The codes from the individual ACS reports are not unique identifiers. Some reports use the same code as key for different values.
- More information is available in [DATA_STRUCTURE](DATA_STRUCTURE.md).

### 2. Comparison of the different police reports.
Run: police_report_comparison.py

Objective: Compare the police reports from different departments.

Result:
- The police departments use different types of reports.
- The reports contain different features in varying numbers.
- The reports are not comparable with each other.
- More information is availabe in [DATA_STRUCTURE](DATA_STRUCTURE.md).


### 3. Analysis of the data from Boston Police Department
Run: dept_11_analysis_main.py

Objective: Perform an analysis of the data provided by the Boston Police Department.

Result: See the paper for detailed results.

