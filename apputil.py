
import pandas as pd
# Exercise 1: Survival Demographics
# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

def survival_demographics(df):
    """
    Analyze Titanic survival patterns by passenger class, sex, and age group.
    
    Returns a DataFrame with columns:
    Pclass, Sex, age_group, n_passengers, n_survivors, survival_rate
    """

    # 1. Create age groups
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
    df['age_group'] = df['age_group'].astype('category')

    # 2. Group by Pclass, Sex, age_group
    grouped = df.groupby(['Pclass', 'Sex', 'age_group'], observed=False)

    # 3. Aggregate counts and calculate survival rate
    summary = grouped['Survived'].agg(
        n_passengers='count',
        n_survivors='sum'
    ).reset_index()
    summary['survival_rate'] = (summary['n_survivors'] / summary['n_passengers'] * 100).round(2)

    # 4. Ensure all combinations are present
    all_combinations = pd.MultiIndex.from_product(
        [sorted(df['Pclass'].unique()), df['Sex'].unique(), age_labels],
        names=['Pclass', 'Sex', 'age_group']
    )
    summary = summary.set_index(['Pclass', 'Sex', 'age_group'])
    summary = summary.reindex(all_combinations, fill_value=0).reset_index()

    # Sort results for readability
    summary = summary.sort_values(by=['Pclass', 'Sex', 'age_group']).reset_index(drop=True)

    return summary


# Visualization function for Streamlit
import plotly.express as px
def visualize_demographic():
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
def family_groups(df):
    """
    Group Titanic passengers by family size and class, and calculate fare statistics.

    Args:
        df (pd.DataFrame): Titanic dataset.

    Returns:
        pd.DataFrame: Table grouped by family_size and Pclass, with columns:
            - n_passengers: Number of passengers in each group
            - avg_fare: Average fare for the group
            - min_fare: Minimum fare in the group
            - max_fare: Maximum fare in the group
    """
    # Create a new column for family size (siblings/spouses + parents/children + self)
    df = df.copy()
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    # Group by family size and class, then aggregate fare statistics
    grouped = df.groupby(['family_size', 'Pclass'], observed=False)
    summary = grouped['Fare'].agg(
        n_passengers='count',
        avg_fare='mean',
        min_fare='min',
        max_fare='max'
    ).reset_index()
    summary['avg_fare'] = summary['avg_fare'].round(2)
    # Sort for readability
    summary = summary.sort_values(by=['Pclass', 'family_size']).reset_index(drop=True)
    return summary


def last_names(df):
    """
    Returns a pandas Series with last name as index and count as value.
    """
    # Extract last name from the Name column (before the comma)
    last_names = df['Name'].str.extract(r'([A-Za-z]+),')[0]
    # Count occurrences of each last name
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