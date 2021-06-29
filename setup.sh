mkdir -p ~/.streamlit/

echo "\
[theme]
primaryColor="#50cb3c"
backgroundColor="#efede8"
secondaryBackgroundColor="#d8d4c3"
textColor="#424242"
font = "sans serif"
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml