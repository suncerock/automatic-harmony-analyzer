from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.extractor.youtube import YoutubeIE


def download_youtube_audio(url: str, cache_dir: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"
    } or not YoutubeIE.suitable(url):
        raise ValueError("Enter a valid YouTube video URL.")

    video_id = YoutubeIE.extract_id(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"{video_id}.wav"
    if audio_path.is_file():
        return audio_path

    with TemporaryDirectory(dir=cache_dir) as temporary_dir:
        temporary_path = Path(temporary_dir) / f"{video_id}.wav"
        options = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(temporary_dir) / f"{video_id}.%(ext)s"),
            "noplaylist": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        }
        with YoutubeDL(options) as downloader:
            downloader.download([url])
        if not temporary_path.is_file():
            raise FileNotFoundError(f"YouTube audio was not saved to {temporary_path}.")
        temporary_path.replace(audio_path)
    return audio_path