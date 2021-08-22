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
        app = st.sidebar.radio(
            '',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()