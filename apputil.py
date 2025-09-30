
import pandas as pd
# Exercise 1: Survival Demographics
# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
def survival_demographics(df=None):
    """
    Analyze Titanic survival patterns by passenger class, sex, and age group.
    Returns a DataFrame with columns:
        Pclass, Sex, age_group, n_passengers, n_survivors, survival_rate
    Ensures all group combinations are present, even if empty, and age_group is Categorical dtype.
    """
    if df is None:
        df = globals().get('df')
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    df = df.copy()
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
    df['age_group'] = pd.Categorical(df['age_group'], categories=age_labels, ordered=True)

    grouped = df.groupby(['Pclass', 'Sex', 'age_group'], observed=False)
    summary = grouped['Survived'].agg(
        n_passengers='count',
        n_survivors='sum'
    ).reset_index()
    summary['survival_rate'] = (summary['n_survivors'] / summary['n_passengers'] * 100).round(2)

    all_pclass = sorted(df['Pclass'].unique())
    all_sex = sorted(df['Sex'].unique())
    all_age = age_labels
    all_combinations = pd.MultiIndex.from_product(
        [all_pclass, all_sex, all_age],
        names=['Pclass', 'Sex', 'age_group']
    )
    summary = summary.set_index(['Pclass', 'Sex', 'age_group'])
    summary = summary.reindex(all_combinations, fill_value=0).reset_index()
    summary['n_passengers'] = summary['n_passengers'].astype(int)
    summary['n_survivors'] = summary['n_survivors'].astype(int)
    summary['survival_rate'] = summary.apply(
        lambda row: 0.0 if row['n_passengers'] == 0 else row['survival_rate'], axis=1
    )
    summary['age_group'] = pd.Categorical(summary['age_group'], categories=age_labels, ordered=True)
    summary = summary[['Pclass', 'Sex', 'age_group', 'n_passengers', 'n_survivors', 'survival_rate']]
    summary = summary.sort_values(by=['Pclass', 'Sex', 'age_group']).reset_index(drop=True)
    return summary

# Visualization function for Streamlit
import plotly.express as px
def visualize_demographic(df):
    """
    Create a bar chart of survival rate for teens by passenger class and sex.

    Args:
        df (pd.DataFrame): Titanic dataset.

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart figure.
    """
    result = survival_demographics(df)
    teen_data = result[result['age_group'] == 'Teen']
    fig = px.bar(
        teen_data,
        x='Pclass',
        y='survival_rate',
        color='Sex',
        barmode='group',
        title='Survival Rate for Teens by Passenger Class and Sex',
        labels={'survival_rate': 'Survival Rate (%)', 'Pclass': 'Passenger Class'}
    )
    return fig


# Exercise 2: Family Size and Wealth
def family_groups(df=None):
    """
    Group Titanic passengers by family size and class, and calculate fare statistics.
    Returns a DataFrame with columns: family_size, Pclass, n_passengers, avg_fare, min_fare, max_fare
    """
    if df is None:
        df = globals().get('df')
    df = df.copy()
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    grouped = df.groupby(['family_size', 'Pclass'], observed=False)
    summary = grouped['Fare'].agg(
        n_passengers='count',
        avg_fare='mean',
        min_fare='min',
        max_fare='max'
    ).reset_index()
    summary['avg_fare'] = summary['avg_fare'].round(2)
    summary = summary[['family_size', 'Pclass', 'n_passengers', 'avg_fare', 'min_fare', 'max_fare']]
    summary = summary.sort_values(by=['Pclass', 'family_size']).reset_index(drop=True)
    return summary

def last_names(df=None):
    """
    Returns a pandas Series with last name as index and count as value.
    """
    if df is None:
        df = globals().get('df')
    # Extract last name (allow for letters, spaces, apostrophes, hyphens)
    last_names = df['Name'].str.extract(r"([A-Za-z'\- ]+),")[0].str.strip()
    return last_names.value_counts()

# Visualization: Most common families by family size
def visualize_families(df, top_n=10):
    """
    Create a bar chart of the most common families (last names) and their family sizes.

    Args:
        df (pd.DataFrame): Titanic dataset.
        top_n (int): Number of top families to display.

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart figure.
    """
    # Get last name counts as a DataFrame
    last_name_counts = last_names(df).reset_index()
    last_name_counts.columns = ['LastName', 'FamilySize']
    # Select the top N families
    top_families = last_name_counts.head(top_n)
    # Create the bar chart
    fig = px.bar(
        top_families,
        x='LastName',
        y='FamilySize',
        title=f'Top {top_n} Most Common Families on the Titanic',
        labels={'FamilySize': 'Number of Family Members', 'LastName': 'Family (Last Name)'}
    )
    return fig