# Converts the videos to mp3
# after if we have 100 of vedeo in plalist we are doing by loop every file to audio and give name depends on vedo 
#  in these code we split by | if we dont want full NAME ONLY START SO WE ARE DOING EVERY MP4 TP MP3 BY FFMEG VIEDO TO AUDIO AND AFTER 
# i ahve only ine vedo i dont want to use these loop thats thwe logic
import os 
import subprocess

files = os.listdir("videos") 
for file in files: 
    tutorial_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜ ")[0]
    print( tutorial_number,  file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])