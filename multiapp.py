"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.sidebar.markdown('Navigation')
        titles = [a["title"] for a in self.apps]
        functions = [a["function"] for a in self.apps]
        title = st.sidebar.radio(
            label='Go To',
            options=titles)

        functions[titles.index(title)]()