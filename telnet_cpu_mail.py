#coding=utf-8
import telnetlib
import re
import time
import smtplib
from email.mime.text import MIMEText
def cpu():
    while True:
      time.sleep(3600)                                         #每一小时检测一次
      HOST = "10.7.1.252"
      password = 'admin'
      tn = telnetlib.Telnet(HOST)
      tn.read_until(b"Password: ")
      tn.write(password.encode('ascii') + b"\n")            #登陆
      tn.write(b"sh proc cpu\n")                            #输入命令
      tn.write(b'ls\n')
      tn.write(b"exit\n")                                   #退出
      m=re.search('\d+%',tn.read_all().decode('ascii'))     #利用正则表达式获取使用率
      a=m.group()
      b=re.search('\d+',a)
      c=int(b.group())                                       #转化为int型
      print ('目前交换机cpu的使用率%d%%'%c)
      cpu_liyonglv = c
      if cpu_liyonglv >20  :                                  #使用率超过20%时发邮件
              baojing()
def baojing():
      msg_from='*******'                           #发送方邮箱
      passwd='******'                              #填入发送方邮箱的授权码
      msg_to='******'                              #收件人邮箱
      subject="交换机cpu"                                    #标题
      content="交换机CPU过载提醒"                            #正文
      msg = MIMEText(content)
      msg['Subject'] = subject
      msg['From'] = msg_from
      msg['To'] = msg_to
      try:
          s = smtplib.SMTP_SSL("smtp.qq.com",465)
          s.login(msg_from, passwd)
          s.sendmail(msg_from, msg_to, msg.as_string())
          print ("发送交换机报警提示成功")
      except :
          print ("发送失败")
      finally:
           s.quit()
cpu()
