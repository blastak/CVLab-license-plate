from pytubefix import YouTube
from pytubefix.cli import on_progress

url = "https://www.youtube.com/watch?v=WRKXnQWfn_A"

yt = YouTube(url, on_progress_callback=on_progress)
print(yt.title)

# ys = yt.streams.get_highest_resolution() # 이게 잘 안먹힘
# ys.download()

for i,s in enumerate(yt.streams):
    print('[%2d]' % (i + 1), s)
j = int(input('어떤 번호를 원하냐? (숫자 입력 후 엔터) '))
ys = yt.streams[j-1]  # 스트림 중에 하나 고름
print('selected stream : ', str(ys))

# ys.download() # Uncomment to use

