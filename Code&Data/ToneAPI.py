# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:08:49 2019

@author: LENOVO
"""
import pandas as pd
import numpy as np
import urllib
import requests
from bs4 import BeautifulSoup
import json
from watson_developer_cloud import ToneAnalyzerV3

import urllib
import lxml.html
class Song(object):
    def __init__(self, artist, title):
        self.artist = artist
        self.title = title
        self.url = None
        self.lyric = None



    def __quote(self, s):
         return urllib.parse.quote(s.replace(' ', '_'))

    def __make_url(self):
        artist = self.__quote(self.artist)
        title = self.__quote(self.title)
        artist_title = artist+':'+title
        url = 'http://lyrics.wikia.com/' + artist_title
        self.url = url



    def lyricwikia(self):
        self.__make_url()
        doc = lxml.html.parse(self.url)
        #print(type(doc))
        lyricbox = doc.getroot().cssselect('.lyricbox')[0]
        print(type(lyricbox))
        lyrics = []

        for node in lyricbox:
            if node.tag == 'br':

                lyrics.append('\n')
               # print(node.tail)
            if node.tail is not None:
                #print(node.tail)
                lyrics.append(node.tail)

        self.lyric =  "".join(lyrics).strip()
        return self.lyric
        return lyrics


song = Song(artist='Ed Sheeran', title='Shape of You')
lyr = song.lyricwikia()
print(lyr)

tone_analyzer = ToneAnalyzerV3(
    version='2019-02-17',
    iam_apikey='6zSlji48p8DDphjnF_ZgfuU4pyP5PlXCk7LOEZq-YieR',
    url='https://gateway-lon.watsonplatform.net/tone-analyzer/api'
)

##text = 'Team, I know that times are tough! Product '\
  ##  'sales have been disappointing for the past three '\
   # 'quarters. We have a competitive product, but we '\
   # 'need to do a better job of selling it!'

text= lyr



tone_analysis = tone_analyzer.tone(
    {'text': text},
    'application/json'
).get_result()

for i in range(0,len(tone_analysis['document_tone']['tones'])):
    print(tone_analysis['document_tone']['tones'][i]['tone_name'])
print(json.dumps(tone_analysis, indent=2))