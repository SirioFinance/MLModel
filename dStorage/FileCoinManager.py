import io
from lighthouseweb3 import Lighthouse

def retreiveFiles(apikey, file_cid):
    destination_path = "./RawLoans.csv"

    file_info = apikey.download(file_cid)    
    file_content = file_info[0]

    with open(destination_path, 'wb') as destination_file:
            destination_file.write(file_content)

    print("Download successful!")

apikey = Lighthouse(token="7473fcf2.b17566b9f5b04683adc93d4b2ff5b988")

file_cid = "bafkreidu7o4uucptfs6ydbmglx67rplvspc2zimpcxtpxi4qjywjbhkypi"
retreiveFiles(apikey, file_cid)
