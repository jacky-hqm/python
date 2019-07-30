# -*- coding: utf-8 -*-
import urllib
import requests
from bs4 import BeautifulSoup
import json
import lxml

def carwl(id):
    url='https://www.lagou.com/jobs/%s.html'% id
    headers={
        'Host': 'www.lagou.com',
        'Referer': 'https://www.lagou.com/jobs/list_python?labelWords=&fromSearch=true&suginput=',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36',
    }
    req=requests.get(url,headers=headers)
    soup=BeautifulSoup(req.content,'lxml')
    job_bt=soup.find('dd',attrs={'class':'job_bt'})

    return job_bt.text



def main() :
     headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36',
            'Host': 'www.lagou.com',
            'Referer': 'https://www.lagou.com/jobs/list_python?labelWords=&fromSearch=true&suginput=',
            'X-Anit-Forge-Code': '0',
            'X-Anit-Forge-Token': None,
            'X-Requested-With': 'XMLHttpRequest',
            'Cookie':'_ga=GA1.2.308641126.1521080288; _gid=GA1.2.2016971252.1521080288; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1521080288; '
                     'user_trace_token=20180315101808-1a3c9ba4-27f7-11e8-b201-525400f775ce; LGSID=20180315101808-1a3ca21c-27f7-11e8-b201-525400f775ce;'
                     ' PRE_UTM=m_cf_cpc_baidu_pc; PRE_HOST=www.baidu.com; PRE_SITE=https%3A%2F%2Fwww.baidu.com%2Fbaidu.php%3Fsc.0s0000KlmKuCTsdU2url1XTyOB2fC0W6iWuy'
                     '4rE7C89taJrrG2VMz-slCV0hiLcFS0Z114-oNL34SHVHU0ESl0JiIgT_K3nbbJw-hyz8_pgb50xt9B9l0qg6KlT6jYT0zfRj-mPndbNI6s0Z-HhzHJ3gadaw_1diriBkC_LGv_szDok1M0.'
                     'DD_NR2Ar5Od663rj6tJQrGvKD7ZZKNfYYmcgpIQC8xxKfYt_U_DY2yP5Qjo4mTT5QX1BsT8rZoG4XL6mEukmryZZjzdT52h881gE4TMH8Hgo4T5MY3IMo9vUt5MEsethZved2s1f'
                     '_TX1_LUd.U1Yk0ZDqs2v4_sK9uZ745TaV8Un0mywkIjYz0ZKGm1Yk0Zfqs2v4_sKGUHYznWR0u1dBugK1n0KdpHdBmy-bIfKspyfqn1c0mv-b5Hc3n0KVIjYknjDLg1DsnH-xnH0'
                     'zndt1njDdg1nvnjD0pvbqn0KzIjYdPjf0uy-b5HDYn1IxnWDsrjPxnW04nW60mhbqnW0Y0AdW5HTzn10zPHbzP-tLnWnsnWR4n1FxnNtknjFxn0KkTA-b5H00TyPGujYs0ZFMIA'
                     '7M5H00mycqn7ts0ANzu1Ys0ZKs5H00UMus5H08nj0snj0snj00Ugws5H00uAwETjYs0ZFJ5H00uANv5gKW0AuY5H00TA6qn0KET1Ys0AFL5HDs0A4Y5H00TLCq0ZwdT1Y4Pjnkn'
                     'jT4Pj64rjfvP1mvPj0d0ZF-TgfqnHRznH03njc3nWT3PsK1pyfqryFWuAcsn1Dsnj0snARYnsKWTvYqnHndfWczrHfLnj04wW6LffK9m1Yk0ZK85H00TydY5H00Tyd15H00XMf'
                     'qn0KVmdqhThqV5HKxn7tsg1Kxn0Kbmy4dmhNxTAk9Uh-bT1Ysg1Kxn7t1nH6vrHIxn0Ksmgwxuhk9u1Ys0AwWpyfqn0K-IA-b5iYk0A71TAPW5H00IgKGUhPW5H00Tydh5HDv0'
                     'AuWIgfqn0KhXh6qn0Khmgfqn0KlTAkdT1Ys0A7buhk9u1Yk0Akhm1Ys0APzm1Yznjm4n6%26ck%3D7217.6.88.223.146.206.184.345%26shh%3Dwww.baidu.com%26sht'
                     '%3Dbaidu%26us%3D1.0.1.0.1.301.0%26ie%3Dutf-8%26f%3D8%26tn%3Dbaidu%26wd%3D%25E6%258B%2589%25E9%2592%25A9%26rqlang%3Dcn%26inputT%3D2154%2'
                     '6bc%3D110101; PRE_LAND=https%3A%2F%2Fwww.lagou.com%2Flp%2Fhtml%2Fcommon.html%3Futm_source%3Dm_cf_cpc_baidu_pc%26m_kw%3Dbaidu_cpc_hz_e110f9_'
                     '265e1f_%25E6%258B%2589%25E9%2592%25A9; LGUID=20180315101808-1a3ca462-27f7-11e8-b201-525400f775ce; JSESSIONID=ABAAABAAAFCAAEGAB43C928F33D3E'
                     '07323DAE8EB9181A6E; index_location_city=%E6%9D%AD%E5%B7%9E; hideSliderBanner20180305WithTopBannerC=1; TG-TRACK-CODE=index_search; LGRID=2'
                     '0180315101830-2718512d-27f7-11e8-b201-525400f775ce; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1521080313; SEARCH_ID=3ed8abc6feec4b19b45ad9486bc1a47a',

              }


     positions=[]
     for x in range(1,2):
          data={
            'first': 'true',
            'pn': x,
            'kd': 'python'
           }

          result=requests.post('https://www.lagou.com/jobs/positionAjax.json?city=%E6%9D%AD%E5%B7%9E&needAddtionalResult=false&isSchoolJob=0',
                         headers=headers,data=data)
          json_result=result.json()
          page_positions=json_result['content']['positionResult']['result']
          for position in page_positions:
              position_dict={
                  'position_name':position['positionName'],
                  'work_year':position['workYear'],
                  'salary':position['salary'],
              }
              #position_id=position['positionId']
              #position_detail=carwl(position_id)
              #position_dict['position_detail']=position_detail
          #positions.extend(position_dict)
          positions.extend(page_positions)

     line=json.dumps(positions,ensure_ascii=False)
     print(line)
     with open('lagou.json','wb+')as fp:
         fp.write(line.encode('utf-8'))


if __name__=='__main__':
    main()
    #carwl(4032193)

