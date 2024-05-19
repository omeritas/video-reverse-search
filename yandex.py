import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import os
import json
import random
import time
import threading
from skimage.metrics import structural_similarity as ssim

# scrape URLs
def scrape_urls(url):
    # Send a request to the URL
    response = requests.get(url)
    response.raise_for_status()  # check if succesful

    # Parsing html
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all image and link elements w/ beautifulsoup
    image_elements = soup.find_all('a', class_='Link Thumb Thumb_hover_fade Thumb_shade Thumb_rounded Thumb_type_inline Thumb_lazy')
    link_elements = soup.find_all('a', class_='Link Link_view_default')

    # Extract URLs
    image_urls = []
    link_urls = []

    for image_element in image_elements:
        if image_element and 'href' in image_element.attrs:
            image_url = image_element['href']
            # if URLs start with //, prepend https to create valid link (result: https://)"
            if image_url.startswith('//'):
                image_url = 'https:' + image_url
            image_urls.append(image_url)
        else:
            image_urls.append('Image URL not found')

    for link_element in link_elements:
        link_url = link_element['href'] if link_element else 'Link URL not found'
        link_urls.append(link_url)

    return image_urls, link_urls

# SSIM checks if the frames are different from the other frames (change the ssim_threshold var to modify) 
# This way we only use the image reverse search engine with lower quantity and more quality frames
def is_significantly_different(prev_frame, current_frame, ssim_threshold=0.8):
    if prev_frame is None:
        return True
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    s, _ = ssim(prev_gray, current_gray, full=True)
    return s < ssim_threshold

# Process the video and save significantly different frames
def process_video(video_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_significantly_different(prev_frame, frame):
            frame_filename = f"{output_directory}/frame_{saved_frame_count}.png"
            cv2.imwrite(frame_filename, frame)
            prev_frame = frame
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    return [f"{output_directory}/frame_{i}.png" for i in range(saved_frame_count)]

# Reverse image search using Yandex
def reverse_image_search(file_path, proxy):
    try:
        search_url = 'https://yandex.ru/images/search'
        proxy_dict = {"http": proxy, "https": proxy}
        files = {'upfile': ('blob', open(file_path, 'rb'), 'image/jpeg')}
        params = {'rpt': 'imageview', 'format': 'json', 'request': '{"blocks":[{"block":"b-page_type_search-by-image__link"}]}'}
        response = requests.post(search_url, params=params, files=files, proxies=proxy_dict)
        query_string = json.loads(response.content)['blocks'][0]['params']['url']
        img_search_url = search_url + '?' + query_string
        return img_search_url
    except requests.exceptions.RequestException as e:
        print(f"Error occurred for {file_path}: {e}")
        return None

# Define the list of proxies to use the reverse image search in a round-robin fashion
proxies = [
    # ex. "http://xxxx@xxxx:8000",
]

# Handle each proxy's operation
def handle_proxy(frame, proxy):
    search_url = reverse_image_search(frame, proxy)
    if search_url:
        proxy_ip = proxy.split('@')[-1].split(':')[0]
        print(f"Search URL for {frame} using proxy {proxy_ip}: {search_url}")

        # Scrape the URLs from the search result page
        image_url, link_url = scrape_urls(search_url)
        print(f"Image URL for {frame}: {image_url}")
        print(f"Link URL for {frame}: {link_url}")

    # Proxy waits for a random interval before continueing
    time.sleep(random.randint(60, 120))

# Specify the video and output and execute the script
def main():
    video_path = 'YOUR-VIDEO-HERE.mp4'
    output_directory = 'extracted_frames'
    extracted_frames = process_video(video_path, output_directory)

    for i, frame in enumerate(extracted_frames):
        proxy_index = i % len(proxies)
        proxy_thread = threading.Thread(target=handle_proxy, args=(frame, proxies[proxy_index]))
        proxy_thread.start()

if __name__ == "__main__":
    main()

