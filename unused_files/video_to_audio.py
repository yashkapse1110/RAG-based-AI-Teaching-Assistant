import os
import subprocess

files = os.listdir("Videos")
for file in files:
    print(file)
    video_no = file.split(".")[0].split("o")[1]
    print(video_no)
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{video_no}.mp3"])

    