#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@version: V1.0
@author: Hazza Cheng
@contact: hazzacheng@gmail.com
@time: 2018/01/24 
@file: douban_client.py 
@description: Login at Douban and change the signature
@modify: 
"""
from html.parser import HTMLParser

import requests


def _attr(attr_list, attr_name):
    for attr in attr_list:
        if attr[0] == attr_name:
            return attr[1]
    return None


class DoubanClient(object):
    def __init__(self):
        object.__init__(self)
        headers = {
            'User-Agent': 'Mozilla / 5.0(X11; Linux x86_64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 63.0.3239.132 Safari / 537.36',
            'Origin': 'https: // www.douban.com'
        }
        self.session = requests.session()
        self.session.headers.update(headers)

    def login(self, username, passwd,
              source='main',
              redir='https://movie.douban.com/cinema/nowplaying/nanjing/',
              login='登录',
              remember='on'):
        url = 'https://accounts.douban.com/login'
        data = {
            'source': source,
            'redir': redir,
            'form_email': username,
            'form_password': passwd,
            'remember': remember,
            'login': login
        }
        headers = {
            'Host': 'accounts.douban.com',
            'Referer': 'https://www.douban.com/accounts/login?source=main'
        }
        self.session.post(url, data=data, headers=headers)
        print(self.session.cookies.items())

    def edit_signature(self, username, signature):
        url = 'https://www.douban.com/people/%s/' % username
        r = self.session.get(url)
        data = {
            'ck': _get_ck(r.content),
            'signature': signature
        }
        edit_signature_url = 'https://www.douban.com/j/people/%s/edit_signature' % username
        headers = {
            'Host': 'www.douban.com',
            'Referer': url,
            'X-Requested-With': 'XMLHttpRequest'
        }
        r = self.session.post(edit_signature_url, data=data, headers=headers)
        print(r.content)


def _get_ck(content):

    class CKParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.ck = None

        def handle_starttag(self, tag, attrs):
            for attr in attrs:
                if tag == 'input' and _attr(attrs, 'type') == 'hidden' and _attr(attrs, 'name') == 'ck':
                    self.ck = _attr(attrs, 'value')

    ckp = CKParser()
    ckp.feed(content.decode('utf-8'))
    return ckp.ck


if __name__ == '__main__':
    dc = DoubanClient()
    dc.login('Your Email', 'Your PassWord')
    dc.edit_signature("Your Signature", 'Hello Spider!')
