import requests
from bs4  import BeautifulSoup
import os
import urllib

BASE_list='http://www.doutula.com/photo/list/?page='
PAGE_list=[]
for x in range(1,10):
    url1=BASE_list+str(x)
    PAGE_list.append(url1)

def download_image(url):
    split_list=url.split('/').pop().split('!')[0]
    print split_list
    filename=split_list
    path=os.path.join('image',filename)
    urllib.urlretrieve(url,filename=path)

def get_page(page_url):
    response=requests.get(page_url)
    content=response.content
    soup=BeautifulSoup(content,'lxml')
    img_list=soup.find_all('img',attrs={'class':'img-responsive lazy image_dta'})
    for img in img_list:
        url=img['data-original']
        print url
        download_image(url)

def main():
    for page_url in PAGE_list:
        get_page(page_url)
if __name__=='__main__':
    main()
