import streamlit as st

def app():
    st.title('FORGE Team Members')
    st.markdown('---')
    author1, _, author2 = st.columns((3,1,3))
    st.markdown('<br>', unsafe_allow_html=True)
    author1.image(r'./assets/author1.png')
    author1.markdown(f"""<center>
        <h2>Kenneth A. Quisado</h2>
        kaquisado@gmail.com<br>
        <a href="https://www.linkedin.com/in/kaquisado/">LinkedIn</a> <a href="https://github.com/kenquix">GitHub</a></center>
    """, unsafe_allow_html=True)
    author1.markdown('---')
    author1.markdown(f"""
        <center>Remote sensing. Python. Cat person.</center><br><br>
    """, unsafe_allow_html=True)

    author2.image(r'./assets/author2.png')
    author2.markdown(f"""
        <center><h2>Ma. Verlina E. Tonga</h2>
        foresterverlinatonga@gmail.com<br>
        <a href="https://www.linkedin.com/in/ma-verlina-tonga-444562a4/">LinkedIn</a> <a href="https://github.com/kenquix">GitHub</a>
    </center>""", unsafe_allow_html=True)
    author2.markdown('---')
    author2.markdown(f"""
        <center>Forester. Environment Planner.</center>
    """, unsafe_allow_html=True)