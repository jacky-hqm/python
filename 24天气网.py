# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import time
from echarts import Echart, Legend, Bar, Axis

temperature_list=[]
city_list=[]
max_list=[]
def get_temperature(url):

    headers={'Referer': 'http://www.weather.com.cn/forecast/',
         'Host': 'www.weather.com.cn',
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36',
         'Upgrade-Insecure-Requests': '1',

         }
    req=requests.get(url,headers=headers)
    soup=BeautifulSoup(req.content,'lxml')
    conMidtab=soup.find('div',class_='conMidtab')
    conMidtab2_list=conMidtab.find_all('div',class_='conMidtab2')
    for x in conMidtab2_list:
        tr_list=x.find_all('tr')[2:]
        for index,tr in enumerate(tr_list):
            if index==0:
                td_list=tr.find_all('td')
                province=td_list[0].text.replace('\n','')
                city=td_list[1].text.replace('\n','')
                temperature=td_list[7].text.replace('\n','')
            else:
                td_list=tr.find_all('td')
                city=td_list[0].text.replace('\n','')
                temperature=td_list[6].text.replace('\n','')

            #print province+city+u' 最高温 '+temperature
            temperature_list.append({
                'city':province+city,
                'max':temperature
            })
            city_list.append(province+city)
            max_list.append(temperature)


def main():
    urls=['http://www.weather.com.cn/textFC/hb.shtml',
          'http://www.weather.com.cn/textFC/db.shtml',
          'http://www.weather.com.cn/textFC/hd.shtml']
    for url in urls:
        get_temperature(url)

    #top10_max=max_list[0:10]
    #top10_cit=city_list[0:10]
    sorted_temprature_list=sorted(temperature_list,lambda x,y:cmp(int(x['max']),int(y['max'])))
    top10_temprature_list=sorted_temprature_list[0:8]
    top10_city_list=[]
    top10_max_list=[]
    for city_max in top10_temprature_list:
        top10_city_list.append(city_max['city'])
        top10_max_list.append(city_max['max'])
        print city_max['city'],city_max['max']

    echart=Echart(u'全国最高温度',u'3.19号')
    bar=Bar(u'最高温度',top10_max_list)
    axis=Axis('category','bottom',data=top10_city_list)
    echart.use(bar)
    echart.use(axis)
    echart.plot()


if __name__=='__main__':
    main()
