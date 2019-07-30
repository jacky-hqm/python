import requests
from bs4  import BeautifulSoup
import os
import urllib
import threading

BASE_list='http://www.doutula.com/photo/list/?page='
PAGE_list=[]
FACE_list=[]
gLock=threading.Lock()
for x in range(1,10):
    url1=BASE_list+str(x)
    PAGE_list.append(url1)


def procuder():
    while True:
        gLock.acquire()
        if len(PAGE_list)==0:
            gLock.release()
            break
        else:
            page_url=PAGE_list.pop()
            gLock.release()
            response=requests.get(page_url)
            content=response.content
            soup=BeautifulSoup(content,'lxml')
            img_list=soup.find_all('img',attrs={'class':'img-responsive lazy image_dta'})
            gLock.acquire()
            for img in img_list:
                url=img['data-original']
                FACE_list.append(url)
            gLock.release()

def customer():
    while True:
      gLock.acquire()
      if len(FACE_list)==0:
        gLock.release()
        continue
      else:
          face_url=FACE_list.pop()
          gLock.release()
          split_list=face_url.split('/').pop().split('!')[0]
          print split_list
          filename=split_list
          path=os.path.join('image',filename)
          urllib.urlretrieve(face_url,filename=path)

def main():
    for x in range(3):
        th=threading.Thread(target=procuder)
        th.start()
    for x in range(5):
        th=threading.Thread(target=customer)
        th.start()

if __name__=='__main__':
    main()
