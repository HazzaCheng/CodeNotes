#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@version: V1.0
@author: Hazza Cheng
@contact: hazzacheng@gmail.com
@time: 2018/01/23 
@file: douban_nowplaying_movies_spider.py
@description: 
@modify: 
"""
import json
import operator
import shutil
import socket
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from html.parser import HTMLParser
import re

import os
from smtplib import SMTP_SSL

import requests
import time

FILEPATH = 'img_temp'

class MovieParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.movies = []
        self.in_movie = False
        self.release_in_movie = False

    def handle_starttag(self, tag, attrs):
        def _attr(attr_list, attr_name):
            for attr in attr_list:
                if attr[0] == attr_name:
                    return attr[1]
            return None

        if tag == 'li' and _attr(attrs, 'data-title') and _attr(attrs, 'data-category') == 'upcoming':
            movie = {}
            movie['title'] = _attr(attrs, 'data-title')
            movie['wish'] = _attr(attrs, 'data-wish')
            movie['duration'] = _attr(attrs, 'data-duration')
            movie['region'] = _attr(attrs, 'data-region')
            movie['director'] = _attr(attrs, 'data-director')
            movie['actors'] = re.split('\s+/\s+', str(_attr(attrs, 'data-actors')))
            self.movies.append(movie)
            self.in_movie = True
        if tag == 'img' and self.in_movie:
            movie = self.movies[-1]
            movie['cover_url'] = _attr(attrs, 'src')
            _download_poster_cover(movie)
        if self.in_movie and tag == 'li' and _attr(attrs, 'class') == 'release-date':
            self.in_movie = False
            self.release_in_movie = True

    def handle_data(self, data):
        if self.release_in_movie:
            self.release_in_movie = False
            movie = self.movies[-1]
            formatted_data = re.compile('\d{2}月\d{2}日').search(data).group()
            movie['release'] = formatted_data


def _download_poster_cover(movie):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.73 Safari/537.36'}
    url = movie['cover_url']
    print('downloading the poster cover of %s from %s' % (movie['title'], url))
    s = requests.get(url, headers=headers)
    fname = FILEPATH + '/' + movie['title'] + '.' + url.split('/')[-1].split('.')[-1]
    with open(fname, 'wb') as f:
        f.write(s.content)
    movie['cover_file'] = fname


def upcoming_movies(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.73 Safari/537.36'}
    res = requests.get(url, headers=headers)
    parser = MovieParser()
    parser.feed(res.content.decode('UTF-8'))
    res.close()
    return parser.movies


def print_movies(movies):
    # movies = sorted(movies, key=operator.itemgetter('rate'), reverse=True)
    comp_func = operator.itemgetter('wish')
    movies.sort(key=comp_func, reverse=True)
    for movie in movies:
        print('=' * 80)
        print('片名：　%s' % movie['title'])
        print('上映日期：　%s' % movie['release'])
        print('多少人想看：　%s' % movie['wish'])
        print('时长：　%s' % movie['duration'])
        print('国家地区：　%s' % movie['region'])
        print('导演：　%s' % movie['director'])
        print('演员： %s' % ' '.join(movie['actors']))
        print('=' * 80)


def send_email(movies, to_addr, to_user):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    from_server = 'smtp.xx.com'    # the smtp server of sender
    from_addr = 'xxx@xxx.com'    # the address of sender
    from_pass = 'xxxxx'   # the passwd of sender
    from_user = 'xxxxx'     # the nickname of sender

    print('Start to send email to %s' % to_addr)
    mail_title = '豆瓣即将上映电影　%s' % time.strftime('%Y-%m-%d', time.localtime(time.time()))
    mail_body = ''
    for movie in movies:
        mail_body += '<p>' + '=' * 8 + '</p>'
        mail_body += '<p>片名：　%s</p>' % movie['title']
        mail_body += '<p>上映日期：　%s</p>' % movie['release']
        mail_body += '<p>多少人想看：　%s</p>' % movie['wish']
        mail_body += '<p>时长：　%s</p>' % movie['duration']
        mail_body += '<p>国家地区：　%s</p>' % movie['region']
        mail_body += '<p>导演：　%s</p>' % movie['director']
        mail_body += '<p>演员： %s</p>' % ' '.join(movie['actors'])
        mail_body += '<p>海报见附件</p>'
        mail_body += '<p><img src="cid:%s"></p>' % movie['cover_file'].split('/')[-1].split('.')[0]
        mail_body += '<p>' + '=' * 8 + '</p>'

    msg = MIMEMultipart('related')
    msg['From'] = _format_addr('%s(%s) <%s>' % (from_user, get_local_ip(), from_addr))
    msg['To'] = _format_addr('%s <%s>' % (to_user, to_addr))
    msg['Subject'] = Header(mail_title, 'utf-8').encode()
    msgAlternative = MIMEMultipart('alternative')
    msg.attach(msgAlternative)
    msg_text = MIMEText(mail_body, 'html', 'utf-8')
    msgAlternative.attach(msg_text)
    # insert images
    for movie in movies:
        with open(movie['cover_file'], 'rb') as f:
            msg_image = MIMEImage(f.read())
        msg_image.add_header('Content-ID', '<%s>' % movie['cover_file'].split('/')[-1].split('.')[0])
        msg.attach(msg_image)
    # ssl login
    smtp = SMTP_SSL(from_server)
    smtp.set_debuglevel(0)
    smtp.ehlo(from_server)
    smtp.login(from_addr, from_pass)
    # send email
    smtp.sendmail(from_addr, to_addr, msg.as_string())
    smtp.quit()
    print('Send email to %s successfully' % to_addr)


def get_local_ip():
    """
    get the local ip
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


if __name__ == '__main__':
    if os.path.exists(FILEPATH):
        shutil.rmtree(FILEPATH)
    os.mkdir(FILEPATH)
    url = r'https://movie.douban.com/cinema/nowplaying/nanjing/'
    movies = upcoming_movies(url)
    print_movies(movies)
    send_email(movies, 'xxxx@xxx.com', 'xxxxx')
    if os.path.exists(FILEPATH):
        shutil.rmtree(FILEPATH)

