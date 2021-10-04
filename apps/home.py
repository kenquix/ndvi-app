import streamlit as st


def app():
    st.markdown(
        f"""<h1><a href="https://sparta.dap.edu.ph/opendata/lgu/butuancity/challenges/butuancity-forest-ecosystem">Sparta Hackathon Challenge</a></h1>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.image(r"./assets/header.jpg", use_column_width=True)
    st.markdown(
        f"""
    <h4>Sector : Forest Ecosystem</h4>

    <h4>Theme : Protecting and controlling forest ecosystem using data and technology</h4><br>

    <p align="justify">Butuan City, also known as the Timber City of the South, enriches its potentials towards 
    investing in the richness of its Forestland Ecosystem. As hampered by some illegal activities and exploitations, 
    the City of Butuan recognizes the relevance of data in the development of technological innovations which can 
    provide mechanisms in protecting forestland areas that can support the economic growth and resiliency of the city. 
    To bring its people to one venue for positive engagement and collaborative efforts, the City of Butuan invites 
    ideas, project proposals, and technological innovations to address threatening factors in protecting and conserving 
    the Forestland ecosystem through hackathons. Datasets and other entries collected in this challenge will be used in 
    hackathons to create a pitch project for Butuan that will address their problem in the tourism sector.</p>
        
    <p align="justify"><em>*This challenge supports the following <a href="https://www.ph.undp.org/content/philippines/en/home/sustainable-development-goals2.html">UN SDGs</a> 
    and <a href="https://www.un.org/securitycouncil/content/repertoire/thematic-items">Thematic issues.</a></p></em></p>
    """,
        unsafe_allow_html=True,
    )

    st.image(r"./assets/SDGs.png")
