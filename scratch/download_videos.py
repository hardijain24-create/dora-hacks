import requests, zipfile, os

def download_and_extract():
    os.makedirs('scratch/videos', exist_ok=True)
    zip_path = 'scratch/videos.zip'
    
    # Download Colours_1of2.zip
    url = 'https://zenodo.org/record/4010759/files/Colours_1of2.zip?download=1'
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('scratch/videos')
    
    os.remove(zip_path)
    print("Done! Extracted to scratch/videos/")

if __name__ == '__main__':
    download_and_extract()
