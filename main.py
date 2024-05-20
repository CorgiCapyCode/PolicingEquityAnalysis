from metadata_analysis import metadata_analysis_main
from police_report_comparison import police_report_comparison_main
from dept_11_analysis_main import dept_11_analysis_main

if __name__ == "__main__":
    print("Welcome to the analysis file for the Policing Equity reports.")
    print("The dataset can be found here: https://www.kaggle.com/datasets/center-for-policing-equity/data-science-for-good")
    print("What kind of analysis should be done?")
    print("For the analysis of the ACS report metadata type: 1")
    print("For the comparison of the different police reports type: 2")
    print("For the analysis of the BPD type: 3")
    analysis_type = input()
    
    if analysis_type == "1":
        metadata_analysis_main()
        print("Finisehd - please check /metadata_analysis for results.")
    elif analysis_type == "2":
        police_report_comparison_main()
        print("Finished - please check police_report_comparison for results.")
    elif analysis_type == "3":
        dept_11_analysis_main()
        print("Finished - please check /dept_11_analysis for results")
    else:
        print("Invalid number")
