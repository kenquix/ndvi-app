mkdir -p ~/.streamlit/

echo "\
[theme]\n\
primaryColor='#50cb3c'\n\
backgroundColor='#efede8'\n\
secondaryBackgroundColor='#d8d4c3'\n\
textColor='#424242'\n\
font='sans serif'\n\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml