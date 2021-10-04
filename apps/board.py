import streamlit as st
import time
import streamlit.components.v1 as components


def app():
    # with st.expander('Discussion board.', expanded=True):
    st.image(r"./assets/header.jpg", use_column_width=True)
    st.markdown(
        f"""
    Here, individuals can post their ideas, provide feedback on the app and/or share the results of your exploration. Also, 
    the general public is encouraged to use this platform to participate in monitoring forest health and resources.
    """,
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    with st.spinner(text="Loading the discussion board..."):
        components.iframe("https://padlet.com/kaquisado/v9y0rhx2lrf0tilk", height=600)
        time.sleep(5)

    placeholder.success("Done!")
    time.sleep(0.5)
    placeholder.empty()
    # st.balloons()
