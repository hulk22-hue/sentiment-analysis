import requests
from bs4 import BeautifulSoup

def fetch_reviews(movie_name, num_reviews=20):
    # Replace spaces with '+' for the search URL
    query = movie_name.replace(" ", "+")
    search_url = f"https://www.imdb.com/find?q={query}&s=tt&ttype=ft&ref_=fn_ft"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(search_url, headers=headers)
    response.raise_for_status()  # Ensure successful request

    soup = BeautifulSoup(response.content, "html.parser")

    for result in soup.find_all('a'):
        if movie_name.lower() in result.text.lower():
            movie_url = f"https://www.imdb.com{result['href']}"
            print(f"Found movie URL: {movie_url}")
            break
    else:
        print(f"ERROR: Movie '{movie_name}' not found.")  
        return []  

    # Construct the reviews URL correctly
    reviews_url = f"{movie_url.split('?')[0]}reviews/"
    reviews = []  

    while len(reviews) < num_reviews:
        print(f"Fetching reviews from: {reviews_url}")
        response = requests.get(reviews_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch reviews (status code: {response.status_code}): {reviews_url}")
            break  
        
        soup = BeautifulSoup(response.content, "html.parser")
        new_reviews = soup.find_all("div", class_="text show-more__control")
        reviews.extend([review.get_text(strip=True) for review in new_reviews])

        load_more = soup.find("button", class_="ipl-load-more__button")
        if not load_more or len(reviews) >= num_reviews:
            break 
        
        next_page = load_more.get("data-key")
        if not next_page:
            break  
        
        reviews_url = f"{movie_url.split('?')[0]}reviews/_ajax?ref_=undefined&paginationKey={next_page}"

    return reviews[:num_reviews]  
