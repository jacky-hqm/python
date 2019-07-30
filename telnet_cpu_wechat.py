#coding=utf-8
import telnetlib
import re
import time
import itchat

def cpu():
    while True:
      time.sleep(3600)                                         #每一小时检测一次
      HOST = "10.7.1.252"
      password = 'admin'
      tn = telnetlib.Telnet(HOST)
      tn.read_until(b"Password: ")
      tn.write(password.encode('ascii') + b"\n")            #登陆
      tn.write(b"sh proc cpu\n")                            #输入命令
      tn.write(b'ls\n'  )
      tn.write(b"exit\n")                                   #退出
      m=re.search('\d+%',tn.read_all().decode('ascii'))     #利用正则表达式获取使用率
      a=m.group()
      b=re.search('\d+',a)
      global c
      c=int(b.group())                                       #转化为int型
      print ('目前交换机cpu的使用率%d%%'%c)
      cpu_liyonglv = c
      if cpu_liyonglv >1  :                                  #使用率超过20%时发邮件
              baojing()
def baojing():
    def lc():
       print("Finash Login!")
    itchat.auto_login(loginCallback=lc)                       #扫二维码登陆网页微信
    time.sleep(3)
    users = itchat.search_friends(name=u'好友名字')           #发给好友=号后面填  u'好友名字'
    userName = users[0]['UserName']                          #找到发送好友名字
    e=('交换机cpu过载提醒,目前交换机cpu的使用率%d%%'%c)   #发送的信息
    itchat.send_msg(msg=e, toUserName=userName)               #发给自己将userName 改成 None
    print('发送完成')

cpu()
