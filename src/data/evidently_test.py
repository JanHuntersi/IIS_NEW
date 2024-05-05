from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.report import Report
from evidently.tests import *
import sys
import os
import pandas as pd
import warnings


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DATA = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'reference_data.csv'))
CURRENT_DATA = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'current_data.csv'))

def main():

    warnings.filterwarnings("ignore")

    reference_data = pd.read_csv(REFERENCE_DATA)
    current_data = pd.read_csv(CURRENT_DATA)

    data_drift_preset_report = Report(
        metrics=[DataDriftPreset()]
    )

    data_drift_preset_report.run(reference_data=reference_data, current_data=current_data)

    # if reports directory does not exist, create it
    if not os.path.exists(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests')):
        os.makedirs(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests'))
    
    reports_json = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests','data_drift.json'))
    reports_html = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests','data_drift.html'))

    # save reportas json and html
    data_drift_preset_report.save(reports_json)
    data_drift_preset_report.save_html(reports_html)

    print(f"Data drift report successfully saved to file {reports_json} and {reports_html}")

    # Define stavility tests

    tests = TestSuite(tests=[
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=reference_data, current_data=current_data)
    test_results = tests.as_dict()

    #print failed tests if any
    if test_results['summary']['failed_tests']:
        print("Failed tests:")
        for test in test_results['failed_tests']:
            print(test)
        sys.exit(1)
    else:
        print("All tests passed!")


    # if reports directory does not exist, create it
    if not os.path.exists(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests','sites')):
        os.makedirs(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests','sites'))
    
    tests_dir = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'reports','evidently_tests','sites','stability_tests.html'))

    tests.save_html(tests_dir)


if __name__ == '__main__':
    main()
