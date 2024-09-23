import pandas as pd

CATEGORY_COLUMN = 'CATEGORY'
TEXT_COLUMN = 'TEXT'


def generate_light_data(df, categories, nb_samples=1000):
    final_df = pd.DataFrame(columns=df.columns)

    for category in categories:
        df_cat = df[df[CATEGORY_COLUMN] == category]
        df_cat = df_cat.head(nb_samples)
        final_df = pd.concat([final_df, df_cat])

    return final_df.sample(frac=1)
