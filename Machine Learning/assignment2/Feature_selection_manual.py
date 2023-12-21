from data_process3 import deal_data
import pandas as pd


def select_feature(file_name):
    # 导入数据
    df = deal_data(file_name)
    # 设置y值
    columns = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'Age', 'ER', 'PgR', 'HER2', 'TrippleNegative',
               'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage']

    return pd.DataFrame(df, columns=columns)
