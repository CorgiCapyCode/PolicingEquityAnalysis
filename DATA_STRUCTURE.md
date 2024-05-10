# Data Structure

The directories sometimes contain files or directories which are doubled. This will be ignored in belows description.

## Abbreviation

| Abbreviation | Translation | Source |
|--------------|-------------|--------|
| CPE | Center for Policing Equity | https://www.kaggle.com/datasets/center-for-policing-equity/data-science-for-good |
| ACS | American Community Survey | https://www.census.gov/programs-surveys/acs |
| HC | Housing Characteristics | https://www.census.gov/library/publications/1951/dec/hc-5.html
| UOF | Use of Force | https://policingequity.org/images/pdfs-doc/CPE_SoJ_Race-Arrests-UoF_2016-07-08-1130.pdf | 
| Dept | Department | |
| FIO | Field Interrogation and Observation | https://data.boston.gov/dataset/boston-police-department-fio |

## General Notes
Link to ACS: https://data.census.gov/

## Directory Structure

![Data Structure](Data_Structure.jpg)
 
### raw_data (root directory)

- Contains directories for different departments and cpe-data.
- The directories in the cpe-data are the same departments with the same files inside.

### Departments (directories)
- The directories for each department follow the same structure and contain the same type of subdirectories / files.
- Contains directories for shapefiles and ACS data.
- Contains a CSV-file with the data from police.

### Shapefiles (subdirectory)
- Contain geo-data that can be explored with GIS-software.

### ACS Data (subdirectory)
- Contains directories with following categories:
    - education-attainment
    - education-attainment-over-25
    - employment
    - income
    - owner-occupied-housing
    - poverty
    - race-age-sex

## File Structures

### ACS_variable_description.csv (in root directory)
- Contains codes (variable names) and descriptions for each code.
- Example: HC01_VC80: Estimate; RACE - Race alone or in combination...

### Police Data (each department directory)
- The strucutre of the police data files differs from department to department.
- TThe file names vary, though "UOF" is often appears in the filename, indicating the focus on "use of force".

- Feature analysis:
    - The files contain different features, totaling 106 unique feature names.
    - The files contain 8 (Dept_49-00035) and 34 (Dept_11-00091) features.
    - Only one feature is used by all departments (INCIDENT_DATE). This feautre exhibits structures (e.g. date and time, only date, separate feature for time).
    - 64 features are unique to a single department.
    - Shared feature names do not necessarly have the same data structure (e.g. INCIDENT_UNIQUE_IDENTIFIER is shared 9 times, but only two use the same structure).
    - Two departments use two types of identifiers.
    - Location data is stored in various features, but is included in all department reports with different structure and details.

- Correlation of Feature Usage
    - Using unprocessed data, the correlation matrix shows the highest correlation between Dept_24-00013 and Dept_49-00033, with 0.52.
        - Dept_24-00013 uses 13 features, with 4 not used by Dept_49-00033.
        - Dept_49-00033 uses 18 features, with 9 not used by Dept_24-00013.
        - Only 9 features are used in both reports.
        - The average correlation is 0.24.
    - Applying a simple categorization based on feature names increased the correlation coefficient up to 0.82.
        - The categorization included for example:
            - Correcting spelling errors (e.g. SUBJECT_RACT instead of SUBJECT_RACE).
            - Splitting all location features into two categories: LOCATION and GEO_INFORMATION.
            - Captuzring the health status in one feature for the subject and for the officer.
            - Summarizing all use of force types into one feature.
            - Creating one unique identifier.
        - The categorization led to a loss of information.
- It is not practical to compare the police data in detail, given the differences in the structure and content.

### ACS_(x)_metadata.csv (each subdirectory in ACS data)
Execute metadata.py

- The metadata per category of each group are equally named for each department.
- Example: All metadata files from the different departments in the education-attainment directory have the same name: ACS_15_5YR_S1501_metadata.
- The metadata files contain two columns:
    - Code (e.g. HC01_EST_VC02)
    - Description (e.g. Total; Estimate; Population 18 to 24 years)
- The length of the unique value dictionary is different for each feature.
    - Code: 1289 unique entries
    - Description: 2053
    - The different metadata files use the same code for different content. The code is thus not a unique identifier.
    - Example: Code HC01_EST_VC03 -> in 'education-attainment' with the description: Total; Estimate; AGE - 16 to 19 years while in 'employment' described as Total; Estimate; Population 18 to 24 years - Less than high school graduate
- Each group (e.g education-attainment) has the same codes and descriptions over all departments. Meaning that the education-attainment group is equal over all departments.

### ACS_(x)_with_ann.csv (each ACS data directory)
- Contains the variables from metadata (e.g. Code HC01_EST_VC02) as features (columns)
- The annotation files are of differnet length (number of data points) for each department.
- The annotation files are of the same length (number of data points) within each group/category in one department.
