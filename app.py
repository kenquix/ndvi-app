import streamlit as st
from multiapp import MultiApp
from apps import home, approach, prototype, board, team # import your app modules here

st.set_page_config(page_title='Vega Map', page_icon=r'./assets/logo.png')

# remove 'Made with Streamlit' footer MainMenu {visibility: hidden;}
hide_streamlit_style = """
			<style>
			footer {visibility: hidden;}
			</style>
			"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# modify the margins
st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 900px;
            padding-left: 3rem;
            padding-right: 3rem;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
    </style>
    """, unsafe_allow_html=True)

make_map_responsive= """
    <style>
    [title~="st.iframe"] { width: 100%}
    </style>
    """
st.markdown(make_map_responsive, unsafe_allow_html=True)

app = MultiApp()

# Add all your application here
app.add_app("The Challenge", home.app)
app.add_app("The Approach", approach.app)
app.add_app("The Prototype", prototype.app)
app.add_app("Discussion Board", board.app)
app.add_app("The Team", team.app)

# The main app
app.run()