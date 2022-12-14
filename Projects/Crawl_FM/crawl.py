import requests
from bs4 import BeautifulSoup


target = "https://www.fmkorea.com/search.php?mid=hotdeal&category=&listStyle=webzine&search_keyword=%EB%AA%A8%EB%8B%88%ED%84%B0&search_target=title_content"


req = requests.get(target)
html = req.text
is_ok = req.ok
status = req.status_code


if is_ok and status == 200:
    print("success")
