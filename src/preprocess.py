# read data
asthma_df = pd.read_csv("../data/asthma_disease_data.csv")


def to_snake_case(name):
    """Add underscore before uppercase letters (except the first), then lowercase

    Args:
        name (str): old column name

    Returns:
        str: new column name
    """    # 
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

asthma_df.columns = [to_snake_case(c) for c in asthma_df.columns]