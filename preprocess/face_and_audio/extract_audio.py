import os
import subprocess

def extract_audio_from_videos(source_folder, target_folder):

    audio_sample_rate = 16000

    for root, dirs, files in os.walk(source_folder):
        # 获取相对于源文件夹的路径
        relative_path = os.path.relpath(root, source_folder)
        
        # 创建目标文件夹
        target_path = os.path.join(target_folder, relative_path)
        os.makedirs(target_path, exist_ok=True)
        
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mkv'):
                # 提取音频文件名
                audio_filename = os.path.splitext(file)[0] + '.wav'
                
                # 源视频文件路径
                video_path = os.path.join(root, file)
                
                # 目标音频文件路径
                audio_path = os.path.join(target_path, audio_filename)
                
                subprocess.call(["ffmpeg", "-i", video_path, "-vn", "-acodec", 'pcm_s16le', '-ac', str(1), '-ar', str(audio_sample_rate), audio_path])
                
                print(f"提取音频：{audio_path}")

# 示例用法
input_folder = "/datasets/Dynamic-expression/DFEW/Clip/original"
output_folder = "/space/lixin/DFEW/Clip/audio_16k"

extract_audio_from_videos(input_folder, output_folder)

