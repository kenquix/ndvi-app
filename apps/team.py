import streamlit as st


def app():
    st.header("FORGE Team Members")
    st.markdown("---")
    _, author1, _, author2, _ = st.columns((2, 3, 1, 3, 2))
    st.markdown("<br><br>", unsafe_allow_html=True)
    author1.image(r"./assets/author1.png")
    author1.markdown(
        f"""<center>
        <h3>Kenneth A. Quisado</h3>
        kaquisado@gmail.com<br>
        <a href="https://www.linkedin.com/in/kaquisado/">LinkedIn</a> <a href="https://github.com/kenquix">GitHub</a></center>
    """,
        unsafe_allow_html=True,
    )
    author1.markdown("---")
    author1.markdown(
        f"""
        <center>Remote sensing. Python. Cat person.</center><br><br>
    """,
        unsafe_allow_html=True,
    )

    author2.image(r"./assets/author2.png")
    author2.markdown(
        f"""
        <center><h3>Ma. Verlina E. Tonga</h3>
        foresterverlinatonga@gmail.com<br>
        <a href="https://www.linkedin.com/in/ma-verlina-tonga-444562a4/">LinkedIn</a> <a href="https://github.com/kenquix">GitHub</a>
    </center>""",
        unsafe_allow_html=True,
    )
    author2.markdown("---")
    author2.markdown(
        f"""
        <center>Forester. Environment Planner.</center>
    """,
        unsafe_allow_html=True,
    )