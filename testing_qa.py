import pandas as pd


def get_question_list(questions_database_path):
    df = pd.read_excel(questions_database_path + "/" + "question_list.xlsx")
    return df["Question"]
