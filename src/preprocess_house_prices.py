
import pandas as pd
import numpy as np

# ------------------------------------------------------------
#  Mapping dictionaries derived from exploratory analysis
#  (static because they should NOT be re‑computed on the test set)
# ------------------------------------------------------------

bldgtype_mapping = {
    'TwnhsE': 5,  # Highest median SalePrice
    '1Fam': 4,
    'Twnhs': 3,
    'Duplex': 2,
    '2fmCon': 1   # Lowest median SalePrice
}

bsmt_exposure_mapping = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

bsmtcond_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

bsmtfintype_mapping = {
    'None': 0,  # No Basement
    'Unf': 1,   # Unfinished
    'LwQ': 2,   # Low Quality
    'Rec': 3,   # Average Rec Room
    'BLQ': 4,   # Below Average Living Quarters
    'ALQ': 5,   # Average Living Quarters
    'GLQ': 6    # Good Living Quarters
}

bsmtqual_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

exterior_mapping = {
    'ImStucc': 'HighValue',
    'Stone': 'HighValue',
    'CemntBd': 'HighValue',
    'VinylSd': 'MidValue',
    'Plywood': 'MidValue',
    'BrkFace': 'MidValue',
    'HdBoard': 'MidValue',
    'Stucco': 'MidValue',
    'MetalSd': 'LowValue',
    'Wd Sdng': 'LowValue',
    'WdShing': 'LowValue',
    'AsbShng': 'LowValue',
    'CBlock': 'Rare',
    'AsphShn': 'Rare',
    'BrkComm': 'Rare'
}

exterqual_mapping = {
    'Ex': 4,  # Excellent
    'Gd': 3,  # Good
    'TA': 2,  # Typical/Average
    'Fa': 1   # Fair
}

fireplacequ_mapping = {
    'None': 0,  # No Fireplace
    'Po': 1,    # Poor
    'Fa': 2,    # Fair
    'TA': 3,    # Typical/Average
    'Gd': 4,    # Good
    'Ex': 5     # Excellent
}

foundation_mapping = {
    'PConc': 5,  # Highest median SalePrice
    'Other': 4,  # Includes Wood and Stone
    'CBlock': 3,
    'BrkTil': 2,
    'Slab': 1    # Lowest median SalePrice
}

garage_type_mapping = {
    'None': 0,      # No Garage
    'Detchd': 1,    # Detached
    'Attchd': 2,    # Attached
    'Basment': 3,   # Basement Garage
    'BuiltIn': 4,   # Built-In
    'CarPort': 5,   # Car Port
    '2Types': 6     # More than one type
}

garagefinish_mapping = {
    'None': 0,  # No Garage
    'Unf': 1,   # Unfinished
    'RFn': 2,   # Rough Finished
    'Fin': 3    # Finished
}

heatingqc_mapping = {
    'Ex': 5,  # Excellent
    'Gd': 4,  # Good
    'TA': 3,  # Typical/Average
    'Fa': 2,  # Fair
    'Po': 1   # Poor
}

housestyle_mapping = {
    '2.5Fin': 8,  # Highest median SalePrice
    '2Story': 7,
    '1Story': 6,
    '1.5Fin': 5,
    'SLvl': 4,
    'SFoyer': 3,
    '2.5Unf': 2,
    '1.5Unf': 1   # Lowest median SalePrice
}

kitchenqual_mapping = {
    'Ex': 5,  # Excellent
    'Gd': 4,  # Good
    'TA': 3,  # Typical/Average
    'Fa': 2,  # Fair
    'Po': 1   # Poor
}

landcontour_mapping = {'Lvl': 3, 'HLS': 2, 'Bnk': 1, 'Low': 0}

landslope_mapping = {'Gtl': 2, 'Mod': 1, 'Sev': 0}

lotshape_mapping = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}

masvnr_mapping = {
    'None': 0,
    'BrkCmn': 1,  # Brick Common
    'BrkFace': 2, # Brick Face
    'Stone': 3    # Stone
}

masvnrtype_mapping = {
    'Stone': 3,  # Highest median SalePrice
    'BrkFace': 2,
    'Other': 1,  # Includes BrkCmn and CBlock
    'None': 0    # No masonry veneer
}

# neighborhood_mapping = neighborhood_stats.rank(ascending=True).to_dict()

roofstyle_mapping = {
    'Shed': 6,      # Highest median SalePrice
    'Flat': 5,
    'Hip': 4,
    'Mansard': 3,
    'Gable': 2,
    'Gambrel': 1    # Lowest median SalePrice
}

#  Columns that were removed from the training set after feature
#  engineering (they will also be removed from the test set)
DROP_COLS = ['1stFlrSF', '2ndFlrSF', 'Alley', 'BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'Exterior1st', 'Exterior1st_Grouped', 'Exterior2nd', 'Exterior2nd_Grouped', 'Fence', 'FullBath', 'Functional', 'GarageCond', 'GarageEfficiency', 'GarageQual', 'GarageType', 'GarageYrBlt', 'HalfBath', 'Heating', 'Id', 'KitchenAbvGr', 'LotArea', 'LotConfig', 'LowQualFinSF', 'MSZoning', 'MiscFeature', 'MiscVal', 'MoSold', 'OpenPorchSF', 'PavedDrive', 'PoolQC', 'RoofMatl', 'SalePrice', 'Street', 'TotRmsAbvGrd', 'Utilities', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold','TotRmsAbvGrd', 'TotalBath','EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea','SaleCondition', 'SaleType']

def preprocess_house_prices(df: pd.DataFrame, stats: dict | None = None):
    """Preprocess the Ames House‑Prices data.

    The function behaves a bit like ``sklearn`` transformers:

    * **Fit + transform** – Call with ``stats=None`` (default) on the
      *training* DataFrame.  The function will
      (1) learn dataset‑specific statistics (such as the mean ``LotFrontage``
      and the neighbourhood ranking) and
      (2) return the transformed DataFrame **together with** the dictionary
      of learned statistics.

    * **Transform only** – Call with the *same* ``stats`` dictionary on any
      subsequent DataFrame (e.g. the Kaggle *test.csv*).  The function will
      perform exactly the same preprocessing using the stored statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame read from ``train.csv`` or ``test.csv``.
    stats : dict | None, optional
        Internal state returned by an earlier call on the training data.

    Returns
    -------
    df_out : pd.DataFrame
        The processed DataFrame, ready for modelling / submission.
    stats : dict
        Dictionary containing all values that must be reused on unseen data.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Fit phase – collect numbers that must be reused later
    # --------------------------------------------------------
    fitting = stats is None
    if fitting:
        stats = {}

        # *LotFrontage* – fill missing with the **training mean**
        stats['lotfrontage_mean'] = df['LotFrontage'].mean()

        # *Neighborhood* – ordinal encode by the median sale price ranking
        if 'SalePrice' in df.columns:
            neighborhood_stats = df.groupby('Neighborhood')['SalePrice'].median()
            stats['neighborhood_mapping'] = neighborhood_stats.rank(
                ascending=True).to_dict()
        else:
            raise ValueError(
                "SalePrice column missing – cannot establish neighbourhood ranking.")

    # --------------------------------------------------------
    # Re‑use stored values during transform phase
    # --------------------------------------------------------
    # Continuous NA imputation
    df['LotFrontage'] = df['LotFrontage'].fillna(stats['lotfrontage_mean'])

    # Categorical NA flags
    for col in ['FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond',
                'BsmtFinType1', 'BsmtFinType2',
                'MasVnrType']:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # 0‑fillers
    for col in ['MasVnrArea', 'GarageYrBlt', 'BsmtQualityScore']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Ordinal / grouped encodings
    mapping_pairs = [
        ('LotShape', lotshape_mapping),
        ('LandContour', landcontour_mapping),
        ('LandSlope', landslope_mapping),
        ('BldgType', bldgtype_mapping),
        ('HouseStyle', housestyle_mapping),
        ('RoofStyle', roofstyle_mapping),
        ('ExterQual', exterqual_mapping),
        ('Foundation', foundation_mapping),
        ('BsmtQual', bsmtqual_mapping),
        ('BsmtCond', bsmtcond_mapping),
        ('BsmtExposure', bsmt_exposure_mapping),
        ('BsmtFinType1', bsmtfintype_mapping),
        ('BsmtFinType2', bsmtfintype_mapping),
        ('HeatingQC', heatingqc_mapping),
        ('KitchenQual', kitchenqual_mapping),
        ('FireplaceQu', fireplacequ_mapping),
        ('GarageType', garage_type_mapping),
        ('GarageFinish', garagefinish_mapping),
        # Exterior grouping creates *new* features – keep originals for now
        ('MasVnrType', masvnrtype_mapping),
    ]

    # Dynamic neighbourhood ranking
    mapping_pairs.append(('Neighborhood', stats['neighborhood_mapping']))

    for col, mapping in mapping_pairs:
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Exterior high‑/mid‑/low value grouping
    for col in ['Exterior1st', 'Exterior2nd']:
        if col in df.columns:
            df[f'{col}_Grouped'] = df[col].map(exterior_mapping)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    if all(c in df.columns for c in ['WoodDeckSF', 'OpenPorchSF']):
        # Create TotalOutdoorSF feature
        df['TotalOutdoorSF'] = df['WoodDeckSF'] + df['OpenPorchSF']

        # Cap outliers at the 99th percentile
        upper_limit = df['TotalOutdoorSF'].quantile(0.99)
        df['TotalOutdoorSF'] = df['TotalOutdoorSF'].clip(upper=upper_limit)

    if 'GarageArea' in df.columns:
        # Cap outliers at the 99th percentile
        upper_limit = df['GarageArea'].quantile(0.99)
        df['GarageArea'] = df['GarageArea'].clip(upper=upper_limit)

    if 'GarageCars' in df.columns:
        # Cap outliers at the 99th percentile
        upper_limit = df['GarageCars'].quantile(0.99)
        df['GarageCars'] = df['GarageCars'].clip(upper=upper_limit)

    if 'Fireplaces' in df.columns:
        # Cap outliers at the 99th percentile
        upper_limit = df['Fireplaces'].quantile(0.99)
        df['Fireplaces'] = df['Fireplaces'].clip(upper=upper_limit)

    if all(c in df.columns for c in ['TotRmsAbvGrd', 'TotalBath']):
    # Create TotalUsableRooms feature
        df['TotalUsableRooms'] = df['TotRmsAbvGrd'] + df['TotalBath']

    if all(c in df.columns for c in ['BsmtFullBath', 'FullBath', 'HalfBath']):
        # Create TotalBath feature
        df['TotalBath'] = (
            df['BsmtFullBath'] + 
            df['FullBath'] + 
            0.5 * df['HalfBath']
        )

        # Cap outliers at the 99th percentile
        upper_limit = df['TotalBath'].quantile(0.99)
        df['TotalBath'] = df['TotalBath'].clip(upper=upper_limit)

    if 'GrLivArea' in df.columns:
        # Cap outliers at the 99th percentile
        upper_limit = df['GrLivArea'].quantile(0.99)
        df['GrLivArea'] = df['GrLivArea'].clip(upper=upper_limit)

    if all(c in df.columns for c in ['1stFlrSF', '2ndFlrSF']):
        # Combine 1stFlrSF and 2ndFlrSF into TotalFlrSF
        df['TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']

        # Cap outliers at the 99th percentile
        upper_limit_total = df['TotalFlrSF'].quantile(0.99)
        df['TotalFlrSF'] = df['TotalFlrSF'].clip(upper=upper_limit_total)

        # Drop the original columns
        df.drop(['1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

    if 'LotArea' in df.columns:
        df['LogLotArea'] = np.log1p(df['LotArea'])

    if all(c in df.columns for c in ['YrSold', 'YearBuilt']):
        df['BuildingAge'] = df['YrSold'] - df['YearBuilt']

    if all(c in df.columns for c in ['YrSold', 'YearRemodAdd']):
        df['TimeSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

    # --------------------------------------------------------
    # Clean‑up – remove columns that the original notebook discarded
    # --------------------------------------------------------
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # --------------------------------------------------------
    # One‑hot encoding
    # --------------------------------------------------------
    for col in ['MSSubClass','MSZoning']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Ensure deterministic column order (useful when saving)
    df = df.sort_index(axis=1)

    return df, stats
