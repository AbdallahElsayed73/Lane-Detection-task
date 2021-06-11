import pytube
url = 'https://www.youtube.com/watch?v=2CIxM7x-Clc&feature=youtu.be'
youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
video.download()