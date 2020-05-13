from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import pyrebase
import time
from wordcloud import WordCloud
import re
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
import wave

import kivy
import pyaudio

kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import json
from ibm_watson import ToneAnalyzerV3

import urllib
import lxml.html
import os, sys, json
from acrcloud.recognizer import ACRCloudRecognizer

import requests
from bs4 import BeautifulSoup
import re
import urllib
import lxml.html
import os, sys, json
import pandas as pd
import numpy as np
import urllib
from ibm_watson import ToneAnalyzerV3
from sklearn.preprocessing import MinMaxScaler
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import NMF



final_res=""
email = 'abc@gmail'
password = 'abcde123'
list1 = []
config = {
	"apiKey": "AIzaSyBqpjwayWXr9GLqKmd7lOquRo1Kgx7LS-Y",
	"authDomain": "lyricquest-89c01.firebaseapp.com",
	"databaseURL": "https://lyricquest-89c01.firebaseio.com",
	"projectId": "lyricquest-89c01",
	"storageBucket": "lyricquest-89c01.appspot.com",
	"messagingSenderId": "72893211179"
}
firebase = pyrebase.initialize_app(config)
#
# auth = firebase.auth()
# user=auth.sign_in_with_email_and_password("abc@gmail.com","abcde123")
#
db = firebase.database()
auth = firebase.auth()

Builder.load_string("""
<MenuScreen>:
	FloatLayout:
		orientation:'vertical'
		Button:	
			text: 'Press to Record'
			size_hint:.9,.1
			pos_hint:{'center_x':0.5,'center_y':0.8}
			on_press:
				root.getAudio()
				root.manager.current='results'           
		Button:
			text: 'Tone Analysis Over Time'
			size_hint:.9,.1
			pos_hint:{'center_x':0.5,'center_y':0.6}
			on_press: root.manager.current = 'analysis'
		Button:
			text: 'User Experience'
			size_hint:.9,.1
			pos_hint:{'center_x':0.5,'center_y':0.4}
			on_press: root.manager.current = 'login_page'
<AnalysisScreen>:
	FloatLayout:
		orientation:'vertical'
		Label:
			pos_hint:{'center_x':0.5,'center_y':0.9}
			font_size:30
			text:"Please enter the year to view its trends"
		TextInput:
			id:year
			pos_hint:{'center_x':0.4,'center_y':0.8}
			size_hint:(0.5,.1)
		Button:
			id:"year"
			text:"Enter"
			pos_hint:{'center_x':0.7,'center_y':0.8}
			size_hint:(0.1,.1)
			on_release:
				root.trends()
		Label:
			pos_hint:{'center_x':0.5,'center_y':0.4}
			font_size:30
			text:"Press the emotion to see its trends across the years"
		Button:
			text:"anger"
			pos_hint:{'center_x':0.2,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"fear"
			pos_hint:{'center_x':0.3,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"joy"
			pos_hint:{'center_x':0.4,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"sadness"
			pos_hint:{'center_x':0.5,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"confident"
			pos_hint:{'center_x':0.6,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"tentative"
			pos_hint:{'center_x':0.7,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"analytical"
			pos_hint:{'center_x':0.8,'center_y':0.3}
			size_hint:(0.1,.1)
			on_release:root.pop_image(self)
		Button:
			text:"Back"
			size_hint:(0.1,0.1)
			pos_hint:{'center_x':0.5,'center_y':0.1}
			on_release:
				root.manager.current="menu"


<ResultsScreen>:
	FloatLayout:
		orientation: 'vertical'
		Label:
			id: result 
			halign: 'center'
			valign: 'top'
			text: root.result
			pos_hint:{'center_x':0.5,'center_y':0.8}

		Button:
			text: 'Result'
			font_size: '30sp'
			size_hint:(0.2,0.1)
			pos_hint:{'center_x':0.5,'center_y':0.5}
			on_press: root.calculate()

		Button:
			text: 'WordCloud'
			font_size: '30sp'
			pos_hint:{'center_x':0.3,'center_y':0.3}
			size_hint:(0.4,0.1)
			on_press: root.wcloud()

		Button:
			text: 'Polarity analysis'
			font_size: '30sp'
			pos_hint:{'center_x':0.7,'center_y':0.3}
			size_hint:(0.4,0.1)
			on_press: root.pop1()
   
		Button:
			text: 'Back to menu'
			
			size_hint:(0.8,0.1)
			size_hint: None, None
			pos_hint:{'center_x':0.5,'center_y':0.1}
			on_press: 
				root.reset()
				root.manager.current = 'menu'


<LoginPage>:
	name:"login_page"

	FloatLayout:
		orientation: 'vertical'
		Label:
			text:"Email ID:"
			pos_hint:{'center_x':0.5,'center_y':.95}	
		TextInput:
			pos_hint:{'center_x':0.5,'top':0.9}
			size_hint:(.9,0.1)
			id: login
		Label:
			text:"Password:"
			pos_hint:{'center_x':0.5,'center_y':0.77}

		TextInput:
			pos_hint:{'center_x':0.5,'top':0.75}
			size_hint:(.9,0.1)

			id: passw
			password: True # hide password
		Button:
			text: "login"
			pos_hint:{'center_x':0.5,'top':0.6}
			size_hint:(.5,0.1)

			on_release: 
				root.sign_in()
				root.pull_playlist()
				root.manager.current="user"
		Label:
			pos_hint:{'center_x':0.5,'top':0.4}
			text: root.exist
			id: error

		Button:
			text:"Back"
			size_hint:(0.5,.1)    
			pos_hint:{'center_x':0.5,'center_y':.1}
			on_release: 
				root.manager.current="menu"


<UserExperience>:
	FloatLayout:
		orientation: 'vertical'	

		ScrollView:
			height:40
			size_hint:(0.5,0.5)
			pos_hint:{'center_x':0.5,'center_y':0.7}
			Label:
				id: Playscroll
				size_hint_y:None
				text_size: self.width, None
				height: self.texture_size[1]
				halign: 'center'
				valign: 'top'
				text: root.playlist	
		Label:
			id: Play
			text: root.playlist
			background_color: 0, 0, 0, 0
			pos_hint:{'center_x':0.5,'top':1}

		TextInput:
			id:SongName
			size_hint:(0.4,0.1)
			pos_hint:{'center_x':.25,'center_y':0.3}
		Button:
			text:"+"
			size_hint:(.1,.1)
			pos_hint:{'center_x':0.5,'center_y':.3}
			on_release:
				root.insert()
		Button:
			text:"-"
			size_hint:(.1,.1)
			pos_hint:{'center_x':0.6,'center_y':.3}
			on_release:
				root.delete()
		Button:
			text:"Analyse"
			size_hint:(.1,.1)
			pos_hint:{'center_x':0.7,'center_y':.3}
			on_release:
				root.recommend()
				root.manager.current="analysis out"


		Button:
			text:"Done"
			size_hint:(0.9,.1)
			pos_hint:{'center_x':0.5,'center_y':.1}
			on_release:
				root.push_playlist()
		Button:
			text:"Show Playlist"
			size_hint:(0.9,.1)    
			pos_hint:{'center_x':0.5,'center_y':.2}
			on_release:root.show()      
		Button:
			text:"Exit"
			size_hint:(0.9,.1)    
			pos_hint:{'center_x':0.5,'center_y':.1}

			on_release:
				root.push_playlist()
				root.manager.current="menu"
<Analysis_out>:
	FloatLayout:
		orientation:"vertical"
		Label:
			text:root.analysisout
			id: answer
			halign: 'center'
			valign: 'top'
			pos_hint:{'center_x':0.5,'center_y':0.9}
		Button:
			text:"Press for result"
			pos_hint:{'center_x':0.5,'center_y':0.5}
			size_hint:(0.5,0.1)
			on_release:
				root.printout()
""")
finalRecommend=[]
new_text=''
title=''
class MenuScreen(Screen):
	def say_hello(self):
		print("Heloooooo")

	def getAudio(self):
		global final_res
		global new_text,title
		CHUNK = 1024
		FORMAT = pyaudio.paInt16
		CHANNELS = 1
		RATE = 44100
		RECORD_SECONDS = 5
		WAVE_OUTPUT_FILENAME = "output.wav"

		p = pyaudio.PyAudio()

		stream = p.open(format=FORMAT,
						channels=CHANNELS,
						rate=RATE,
						input=True,
						frames_per_buffer=CHUNK)

		print("* recording")

		frames = []

		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			frames.append(data)

		print("* done recording")

		stream.stop_stream()
		stream.close()
		p.terminate()

		wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(frames))
		wf.close()



		if __name__ == '__main__':
			config1 = {

				'host': 'identify-eu-west-1.acrcloud.com',
				'access_key': '82034f4000dd1fa4acf2c31d40771f89',
				'access_secret': 'GzybFe6gbI5xeONVJHjfkwpqE4cWSLUv8ZPJM9qx',
				'timeout': 10  # seconds
			}

			'''This module can recognize ACRCloud by most of audio/video file. 
				Audio: mp3, wav, m4a, flac, aac, amr, ape, ogg ...
				Video: mp4, mkv, wmv, flv, ts, avi ...'''
			re = ACRCloudRecognizer(config1)
			# print(sys.argv[1])
#HARD CODED SHAPE OF YOU HEREEE!!!!!
			# recognize by file path, and skip 0 seconds from from the beginning of sys.argv[1].
			#  print(re.recognize_by_file(sys.argv[1], 0))
			try:
				res = re.recognize_by_file('output.wav', 0)
				a = json.loads(res)
			
				title = a['metadata']['music'][0]['title']
				artist_name = a['metadata']['music'][0]['artists'][0]['name']
				print("\n \n Song recognized \n \n")
				self.manager.current='results'
			except Exception as e:
				print("\n Song not recognized \n ")
				return
			# res = re.recognize_by_file("shape.mp3", 0)
			# a = json.loads(res)

			# title = a['metadata']['music'][0]['title']
			# artist_name = a['metadata']['music'][0]['artists'][0]['name']

		import requests
		from requests.auth import HTTPBasicAuth
		from bs4 import BeautifulSoup
		#from requests_ntlm import HttpNtlmAuth
		import re
		url = "http://lyrics.wikia.com/wiki/"+artist_name+":"+title
		page = requests.get(url)
		lyrics = []
		dummy=''
		soup = BeautifulSoup(page.text, 'lxml')

		m = soup.find(class_='lyricbox')

		paragraphs = re.findall(r'<div class="lyricbox">(.*?)<div class="lyricsbreak">', str(soup))
		p = paragraphs[0]

		new_text = re.sub('<br/>', '\n', p)
		print(new_text)

		# tone_analyzer = ToneAnalyzerV3(
		# 	version='2019-02-17',
		# 	iam_apikey='6zSlji48p8DDphjnF_ZgfuU4pyP5PlXCk7LOEZq-YieR',
		# 	url='https://gateway-lon.watsonplatform.net/tone-analyzer/api'
		# )
		from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
		
		authenticator = IAMAuthenticator('07LcRRjqMx-6e7iTD64yqkhCom4mAJIA8tJ9N_s8bUmr')
		tone_analyzer = ToneAnalyzerV3(
			version='2017-09-21',
			authenticator=authenticator
		)
		tone_analyzer.set_service_url('https://api.us-east.tone-analyzer.watson.cloud.ibm.com')
		
		##text = 'Team, I know that times are tough! Product '\
		##  'sales have been disappointing for the past three '\
		# 'quarters. We have a competitive product, but we '\
		# 'need to do a better job of selling it!'

		df=pd.read_csv('lyrics.csv',encoding='utf-8')
		from sklearn.feature_extraction.text import TfidfVectorizer
		stop_words=stopwords.words('english')
		stop_words.extend(['one',
		'yeah',
		'yea',
		'oh',
		'get',
		'got',
		'let',
		'go',
		'would',
		'could',
		'no',
		'for',
		'say',
		'nan',
		'yo',
		'de',
		'ya',
		'em',
		'hey',
		'put',
		'ever',
		'much',
		'got',
		'la',
		'like',
		'cause','chorus','x2','verse','2x','gonna','wanna','gotta','see','said','yes','give','well','tell','make','back','take'])
		stop_words.extend(list(stopwords.words('french')))
		stop_words.extend(list(stopwords.words('spanish')))
		stop_words.extend(list(stopwords.words('danish')))
		df.loc[df['index'] == 1, 'lyrics'] = new_text
		vectorizer=TfidfVectorizer(stop_words=stop_words,min_df=0.04)
		tfidf=vectorizer.fit_transform(df['lyrics'].apply(lambda tfidf: np.str_(tfidf)))
		df.dropna(inplace=True)
		nmf=NMF(n_components=5)
		topic_values=nmf.fit_transform(tfidf)
		topic_labels=[]
		for topic_num, topic in enumerate(nmf.components_): 
			message = "Topic #{}: ".format(topic_num + 1)
			message += " ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-30 :-1]])
			print(message)
			if('know' in message and  'time' in message and  'never' in message and 'life' in message):
				topic_labels.append('Life')
			if('love' in message and 'heart' in message and 'true' in message and 'need' in message ):
				topic_labels.append('Love')
			if('baby' in message and 'want' in message and 'girl' in message and 'know' in message ):
				topic_labels.append('Party')
			if('die' in message and 'hand' in message and 'live' in message and 'life' in message):
				topic_labels.append('Death&Negativity')	
			if('come' in message and 'home' in message and 'us' in message and 'little'):
				topic_labels.append('Travel&Home')	
			message=''

			print(topic_labels)
			print("\n")
			#print([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-30 :-1]])
			#print("\n")
		#topic_labels=['Life','Love','Death&Negativity','Party','Travel&Home']
		df_topics=pd.DataFrame(topic_values, columns=topic_labels)

		#print(df_topics.loc[[1]])
		kk=df_topics.loc[[1]]
		k_dict=list(kk.loc[1])
		n = len(topic_labels)
	 
		# Traverse through all array elements
		for i in range(n):
	 
			# Last i elements are already in place
			for j in range(0, n-i-1):
	 
				# traverse the array from 0 to n-i-1
				# Swap if the element found is greater
				# than the next element
				if k_dict[j] < k_dict[j+1] :
					k_dict[j], k_dict[j+1] = k_dict[j+1], k_dict[j]
					topic_labels[j], topic_labels[j+1] = 	topic_labels[j+1], topic_labels[j]
		print(k_dict)
		print(topic_labels)
		top_labels=topic_labels[0:2]

		top_labels = "\n".join(top_labels)

		text = new_text

		tone_analysis = tone_analyzer.tone(
			{'text': text},
			'application/json'
		).get_result()
		final_res = "Title:" + title + "\n" + "Artist:" + artist_name+"\n"+"The tones identified are:"+'\n'
		#print(tone_analysis)
		for i in range(0, len(tone_analysis['document_tone']['tones'])):
			dummy=dummy+"\n"+tone_analysis['document_tone']['tones'][i]['tone_name']
		if(dummy==""):
			final_res+="\n"+"No dominant tone identified in the song"
		else:
			final_res+="\n"+dummy

		dummy=''
		final_res+='\n'+"The topics extracted are:"+"\n"+top_labels 
		print(final_res)
		pass



class ResultsScreen(Screen):

	result = ''

	def calculate(self):
		global final_res
		self.result=final_res
		self.ids["result"].text = self.result
		print(self.ids["result"].text)



	def reset(self):
		self.ids["result"].text=''


	def wcloud(self):
		global new_text
		lyrics=new_text
		all_words=lyrics
		all_words = re.sub(r'[\(\[].*?[ \)\]]', '', all_words)

		all_words = os.linesep.join([s for s in all_words.splitlines() if s])
		print (all_words)
		word_cloud = WordCloud(width=1000, height=500).generate(all_words.lower())
		word_cloud.to_file("wordcloud.png")
		image = word_cloud.to_image()

		pop = Popup(title='test', content=Image(source='wordcloud.png'), size=(500, 500))
		pop.open()

	def pop1(self):
		global new_text,title

		all_words=new_text
		song_name=title
		all_words = re.sub(r'[\(\[].*?[\)\]]', '', all_words)

		all_words = os.linesep.join([s for s in all_words.splitlines() if s])
		print(all_words)
		f = open('lyrics_for_sent_cleaned.txt', 'wb')
		f.write(all_words.encode('utf-8'))

		df = pd.DataFrame(columns=( 'song_name','pos', 'neu', 'neg'))
		sid = SentimentIntensityAnalyzer()
		i=0
		num_positive = 0
		num_negative = 0
		num_neutral = 0

		f = open('lyrics_for_sent_cleaned.txt', 'rb')
		for sentence in f.readlines():
			this_sentence = sentence.decode('utf-8')
			print(this_sentence)
			print("XXXXXXX")
			comp = sid.polarity_scores(this_sentence)
			comp = comp['compound']
			if comp >= 0.5:
				num_positive += 1
			elif comp > -0.5 and comp < 0.5:
				num_neutral += 1
			else:
				num_negative += 1

		num_total = num_negative + num_neutral + num_positive
		percent_negative = (num_negative / float(num_total)) * 100
		percent_neutral = (num_neutral / float(num_total)) * 100
		percent_positive = (num_positive / float(num_total)) * 100
		df.loc[i] = ( song_name, percent_positive, percent_neutral, percent_negative)
		i+=1

		df.plot.bar(x='song_name', stacked=True)
		
		#plt.show()
		fig1 = plt.gcf()
		#plt.show()
		plt.draw()
		fig1.savefig('plot_testing2.png')
		pop = Popup(title='test', content=Image(source='plot_testing2.png'), size=(500, 500))
		pop.open()

	pass


class AnalysisScreen(Screen):
	def analyze(self):
		print("analyzing")

	def pop_image(self, instance):
		print(instance.text)
		pop = Popup(title='test', content=Image(source=instance.text + '.png'), size=(500, 500))
		pop.open()

	def trends(self):
		year = self.ids["year"].text
		print(year)
		pop = Popup(title='test', content=Image(source=year + '.png'), size=(500, 500))
		pop.open()


class LoginPage(Screen):
	exist = ' '

	def sign_in(self):
		global email, password, obj

		email1 = self.ids["login"].text.split('.')
		email = email1[0]
		password = self.ids["passw"].text
		try:
			user = auth.sign_in_with_email_and_password(email + ".com", password)
		except:

			auth.create_user_with_email_and_password(email + ".com", password)

	def pull_playlist(self):
		global email, password, list1
		try:
			a = list(db.child(email).child("playlist").get().val())[0]
			list1 = list(db.child(email).child("playlist").child(a).get().val())

		except:
			if (len(list1) == 0):
				print(".NO ENTRIES.")

def lyricwikia(title,artist_name):
	url = "http://lyrics.wikia.com/wiki/"+artist_name+":"+title
	page = requests.get(url)
	lyrics = []

	soup = BeautifulSoup(page.text, 'lxml')

	m = soup.find(class_='lyricbox')

	paragraphs = re.findall(r'<div class="lyricbox">(.*?)<div class="lyricsbreak">', str(soup))
	p = paragraphs[0]

	new_text = re.sub('<br/>', '\n', p)
	return new_text


# In[127]:


def get_playlist_tones(playlist):#top_songs):
	play = []
	for i in range(len(playlist)):
	#for i in range(60): #taking the first 60 songs here for the playlist
		artist_name= playlist[i][1]#top_songs.iloc[i, 2]
		title = playlist[i][0]#top_songs.iloc[i, 3]
		try:
			lyr = lyricwikia(title,artist_name)
		except:
			print("\nSONG NOT FOUND\n")
			continue
		tone_analyzer = ToneAnalyzerV3(
			version='2019-02-17',
			iam_apikey='6zSlji48p8DDphjnF_ZgfuU4pyP5PlXCk7LOEZq-YieR',
			url='https://gateway-lon.watsonplatform.net/tone-analyzer/api'
		)
		text = lyr
		if(len(lyr)==0):
			continue
		tone_analysis = tone_analyzer.tone(
			{'text': text},
			'application/json'
		).get_result()
		print(i)
		tone = tone_analysis['document_tone']['tones']
		play.append([artist_name, title, tone])
	return play


# In[128]:


def find_nearest(array, value, n):
	array = np.asarray(array)
	ele=[]
	#print("In nearest")
	for i in range(n):
		#print(i)
		idx = (np.abs(array - value)).argmin()
		if(idx not in ele):
			ele.append(idx)
		else:
			i=i-1
		alist = list(array)
		alist.remove(array[idx])
		array = np.asarray(alist)
	print("NEAREST ELEMENTS TO ZERO:",ele)
	return ele


# In[129]:


def diff_matrix(data, hi_tones):
	hi = np.array(hi_tones)
	diff = data[:,hi[:,0].astype(int)] - hi[:,2].astype(float)
	print("DIFFERENCE:",diff)
	return diff


# In[130]:


def plot_playlist(dkeys, avg_norm_playlist):
	#import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 4))  # width:20, height:3
	plt.bar(dkeys,avg_norm_playlist, align='edge', width=0.7)
	#plt.bar(dkeys,year_t[0], align='center', alpha=0.5)
	plt.xticks(dkeys)
	plt.ylabel('Score')
	plt.xlabel('Tone')
	plt.title("Playlist")
	plt.savefig("fig.png", bbox_inches='tight') #This is the graph right? cool
	#plt.show()

class UserExperience(Screen):
	playlist = ""

	def recommend(self):
		global finalRecommend # ok ok 
		#try:
		if(len(list1)!=0):
			print("Ready to analyse...")
		print(list1)
		play_list = []
		for i in list1:
			temp = i.split(',')
			print(temp)
			play_list.append(temp)
		print(play_list)

		play = get_playlist_tones(play_list) #get formatted play_list+scores
		play_list = pd.DataFrame(play)
		play_list = np.array(play_list)
		zeros_playlist = np.concatenate((play_list,np.zeros([play_list.shape[0],7])),axis = 1)
		dkeys = ['anger', 'fear', 'joy', 'sadness', 'analytical', 'confident']
		for i in range(zeros_playlist.shape[0]):
			sc = zeros_playlist[i][2]
			for j in range(len(sc)):
				if(sc[j]['tone_id'] != 'tentative'):
					ind = dkeys.index(sc[j]['tone_id'])
					zeros_playlist[i][ind+3] = sc[j]['score']
		final_playlist = np.delete(zeros_playlist, [2],1) #final playlist with scores


		# In[144]:


		final_playlist = np.array(final_playlist)
		final_playlist = final_playlist[:,:-1]
		print(final_playlist)


		# In[146]:


		data = pd.read_csv("output_with_song.csv")
		data = data.iloc[:,1:]
		minmax = MinMaxScaler()
		scaled_X = minmax.fit_transform(data.iloc[:, 3:-1])
		scaled_play = minmax.transform(final_playlist[:,2:])
		temp = sum(scaled_play)/scaled_play.shape[0]
		avg_norm = temp.copy()
		sorte = np.sort(avg_norm)
		max3 = sorte[-3:]
		hi_tones = []
		for i in max3:    
			hi_tones.append([list(avg_norm).index(i),dkeys[list(avg_norm).index(i)],i])
		print(hi_tones)


		# In[149]:


		#get_ipython().run_line_magic('matplotlib', 'inline')
		dkeys = ['anger', 'fear', 'joy', 'sadness', 'analytical', 'confident']
		diff = diff_matrix(scaled_X, hi_tones)
		#print(np.sum(diff, axis = 1))
		diff_sums = np.sum(diff, axis = 1)
		nearest_val = find_nearest(diff_sums, 0.0 ,5)
		#print(nearest_val)
		#ind_nearest=[]
		#for i in nearest_val:
		#    ind_nearest.append(list(diff_sums).index(i))
		#print(ind_nearest)
		print(dkeys)
		print("RESULTS:")
		print(avg_norm)
		plot_playlist(dkeys, avg_norm)
		print(data.iloc[nearest_val,1:]) #FINAL RECOMMEND
		#Store the above into global variable already defined
		print(data.iloc[nearest_val,1:3])
		data=pd.DataFrame(data)
		li = list(np.array(data.iloc[nearest_val,1:3]).ravel())
		print("PRINTING LI")
		print(li)
		final = []
		for i in range(0,len(li),2):
		    final.append(str([li[i+1],li[i]]))
		finalRecommend=final
		finalRecommend=["Songs you may like based on your listening"]+["Title, Artist"] +finalRecommend
		for i in range(0,len(finalRecommend)):
			finalRecommend[i]=finalRecommend[i].replace('[',' ')
			finalRecommend[i]=finalRecommend[i].replace(']',' ')

		#n = len(topic_labels)
		print("PRINTING HERE FNAL RECOMMEND")
		print(finalRecommend)
		#except Exception as e:A
			#print("List is empty...")
			#return 

	def show(self):
		global email, password, list1
		self.playlist = "\n".join(list1)
		# print(self.playlist)
		self.ids['Playscroll'].text = self.playlist

	def pull_playlist(self):
		global email, password, list1
		try:
			a = list(db.child(email).child("playlist").get().val())[0]
			list1 = list(db.child(email).child("playlist").child(a).get().val())
			self.playlist = "\n".join(list1)
			# print(self.playlist)
			self.ids['Playscroll'].text = self.playlist
		except:
			if (len(list1) == 0):
				print(".NO ENTRIES.")

	def insert(self):

		global email, password, list1
		#####tone_playlist = self.ids['SongName'].text

		list1.append(self.ids['SongName'].text)
		print("List:", list1)
		self.playlist = "\n".join(list1)
		# print(self.playlist)
		self.ids['Playscroll'].text = self.playlist
		print()
		# db.child("abc@gmail").child("playlist").push("sing2S")

	def delete(self):

		global email, password, list1
		try:
			to_del = self.ids['SongName'].text
			print("Deleting:", list1.remove(to_del))
			self.playlist = "\n".join(list1)
			# print(self.playlist)
			self.ids['Playscroll'].text = self.playlist
		except:
			pass

	def push_playlist(self):

		global email, password, list1,final_res
		firebase = pyrebase.initialize_app(config)
		db = firebase.database()
		db.child(email).remove()
		db.child(email).child("playlist").push(list1)

class Analysis_out(Screen):
	global finalRecommend
	analysisout=''
	def printout(self):
		self.analysisout="\n".join(finalRecommend)
		self.ids['answer'].text=self.analysisout
		pop = Popup(title='test', content=Image(source='fig.png'), size=(500, 500))
		pop.open()



sm = ScreenManager()
sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(ResultsScreen(name='results'))
sm.add_widget(AnalysisScreen(name='analysis'))
sm.add_widget(LoginPage(name='login_page'))
sm.add_widget(UserExperience(name='user'))
sm.add_widget(Analysis_out(name='analysis out'))


class LoginApp(App):
	def build(self):
		Window.size = (600, 600)

		return sm


if __name__ == '__main__':
	LoginApp().run()
