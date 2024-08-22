
import pandas as pd


def classification_report_to_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:  
        row_data = line.split()
        if len(row_data) == 0 or len(row_data) < 5:
            continue
        row = {
            'class': row_data[0],
            'precision': float(row_data[1]),
            'recall': float(row_data[2]),
            'f1-score': float(row_data[3]),
            'support': int(row_data[4])
        }
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe.set_index('class')