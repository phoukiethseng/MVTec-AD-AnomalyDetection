import numpy as np
import pandas as pd
from utils.constants import CATEGORIES

def save_feature_to_excel(file_path, features, category):
    """
    Save extracted features to Excel file. Will overwrite existing category sheet!

    :param overwrite: Overwrite existing data in the Excel sheet
    :param category: data category that data belong to (eg: "bottle", "cable" ...etc.)
    :param file_path: file path to save features to.
    :param features: Numpy array shape of (samples_num, feature_size)
    """

    assert file_path is not None, "file_path cannot be empty"
    assert features is not None, "features cannot be empty"
    assert category is not None, "category cannot be empty"
    assert category in CATEGORIES, "invalid data category"

    print(f"Saving features to Excel file {file_path}")
    new_df = pd.DataFrame(features)

    excel_writer = pd.ExcelWriter(file_path, mode="a", if_sheet_exists="replace")
    new_df.to_excel(excel_writer, sheet_name=category, index=False, header=False)
    excel_writer.close()
    print("Excel file saved successfully.")


def get_feature_from_excel(file_path, feature_size, category="bottle"):
    """
    Get features from saved Excel file

    :param category: Specify which category to get features from (eg: "bottle", "cable" ...etc.). Default: "bottle"
    :param feature_size: number of feature or dimension of feature descriptor
    :param file_path: path to saved feature descriptor
    """
    assert file_path is not None, "file_path cannot be empty"
    assert feature_size > 0, "feature_size cannot be zero"
    assert category in CATEGORIES, "invalid data category"

    select_columns = [col_idx for col_idx in range(feature_size)]
    df = pd.read_excel(file_path, sheet_name=category, header=None, index_col=None, usecols=select_columns)

    X = df.to_numpy()
    return X


def test_get_feature_from_excel():
    file = "../feature_extraction/output/test_hog_descriptor.xlsx"
    get_feature_from_excel(file, feature_size=5, category="bottle", data_class="good")

def test_save_feature_to_excel():
    file = "../feature_extraction/output/test_hog_descriptor.xlsx"
    features = np.ones((4,5))
    save_feature_to_excel(file, features, category="tile", data_class="bad")

if __name__ == '__main__':
    # test_get_feature_from_excel()
    test_save_feature_to_excel()
