import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader.password = input("Password for videos?:\n")
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="./soccernet_download/")

mySoccerNetDownloader.downloadGames(files=["1_HQ.mkv", "2_HQ.mkv"], split=["train","valid","test","challenge"])
