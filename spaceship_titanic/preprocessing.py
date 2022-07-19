import pandas as pd

def drop_columns(df):
    return df.drop(columns=['PassengerId', 'Name'])

def cabin_transform(df):
    # Split 'Cabin' into three separate features
    df[['Cabin_Deck', 'Cabin_Level', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    # Drop original column
    df.drop(columns='Cabin', inplace=True)
    # Cast Cabin_Level from str to float
    df['Cabin_Level'] = pd.to_numeric(df['Cabin_Level'])
    return df

def feat_cols():
    feat_num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    feat_ohe = ['HomePlanet', 'Destination']
    feat_ordstr = ['CryoSleep', 'VIP', 'Cabin_Deck', 'Cabin_Side']
    feat_ordnum = ['Cabin_Level']
    return (feat_num, feat_ohe, feat_ordstr, feat_ordnum)

if __name__ == '__main__':
    pass
