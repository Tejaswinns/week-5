import streamlit as st

from apputil import *

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.write(
'''
# Titanic Visualization 1

'''
)

# Write a clear question for the demographic visualization
st.write("What is the survival rate for teens in all passenger classes?")

# Generate and display the figure
import plotly.express as px
fig1 = visualize_demographic(df)
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Titanic Visualization 2
'''
)
# Write a clear question for the family last name visualization
st.write("Which are the most common families (last names) on the Titanic, and what are their family sizes?")
# Generate and display the figure
fig2 = visualize_families(df)
st.plotly_chart(fig2, use_container_width=True)

st.write(
'''
# Titanic Visualization Bonus
'''
)
# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)