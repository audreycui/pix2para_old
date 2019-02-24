from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from google_images_download import google_images_download

import sys

smith = 'https://americanart.si.edu'

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)

def process(text): #removes html tags and block quote tags
	text = text.encode('ascii', errors = 'ignore')
	text = str(text)

	text = text[text.find(':')+1:]
	ind1 = text.find('<')
	ind2 = text.find('>')
	while (ind1 >= 0 and ind2 >= 0): 
		text = text[0:ind1] + text[ind2+1:]
		ind1 = text.find('<')
		ind2 = text.find('>')

	text = remove(text, '&lt;blockquote&gt;')
	text = remove(text, '&lt;/blockquote&gt;')
	return text

def remove(text, remove): #removes all instances of 'remove' from 'text'
	ind = text.find(remove)
	while (ind>=0): 
		#print("INDEX " + str(ind) )
		text = text[0:ind] + text[ind+len(remove):]
		ind = text.find(remove)
	return text 

def iterateLetters(): 
	base_site = smith + '/art/artists/'
	for i in range (0, 26): 
		letter = chr(i+97)
		#gets html of site with all artists whose last name starts with 'letter'
		html = BeautifulSoup(simple_get(base_site + letter), 'html.parser') 
		has_next_page = html.find('li', attrs = {'class' : 'pager__item pager__item--last'})
		counter = 7
		while (len(has_next_page) > 0): #iterates through all pages under that letter
			if counter == 0: 
				letter_site = base_site + letter
			else:
				letter_site = base_site + letter + '?page=' + str(counter)
			letter_html = BeautifulSoup(simple_get(letter_site), 'html.parser')
			table_col = letter_html.find_all('td', attrs = {'class' : 'priority-low views-field views-field-field-alphabetical-name'})
			for artist in table_col: #iterate through all artists on that page
				artist_site = artist.find('a').get('href')
				print(artist_site)
				scrapeArtist(artist_site)
			has_next_page = html.find('li', attrs = {'class' : 'pager__item pager__item--last'})
			counter += 1

def scrapeArtist(artist_site): 
	artist_site = smith + artist_site
	artist_html = BeautifulSoup(simple_get(artist_site), 'html.parser')
	art_list = artist_html.find('section', attrs = {'class' : 'views-element-container block block-views block-views-blockartists-artwork-artists-artworks-block clearfix'})
	art_list = art_list.find_all('div', attrs = {'class' : 'col col-xs-12 col-sm-12 col-md-6 col-lg-3 '})
	for artwork in art_list: #scrape all of the artwork under that artist
		artwork_site = artwork.find('a').get('href')
		print(artwork_site)
		scrapeArtwork(artwork_site)

def scrapeArtwork(artwork_site):
	artwork_site = smith + artwork_site
	artwork_html = BeautifulSoup(simple_get(artwork_site), 'html.parser')
	caption = artwork_html.find('div', attrs = {'class' : 'field field--name-field-gallery-label field--type-text-long field--label-above'})
	if caption != None: #scrapes gallery label if there is one
	else: #otherwise scrapes exhibition label
		caption = artwork_html.find('div', attrs = {'class' : 'field field--name-field-exhibition-label field--type-text-long field--label-above'})
	if caption != None: 
		caption = caption.find('div', attrs = {'class' : 'field--item'})
		caption = process(caption)
		artwork_title = artwork.html.find('h1' : attrs = {'class' : 'page-header'}) #scrape artwork title
		artwork_title = process(artwork_title)
		print (caption)
		sys.exit(1)


iterateLetters()
