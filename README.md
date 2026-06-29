# Hướng dẫn chạy

docker build -t face-api .
docker run -p 80:80 face-api

API sẽ được chạy trên http://127.0.0.1/

# API

POST /match

Input:
Field Type
image1 Image
image2 Image

Output:  
{  
 "status": "SUCCESS",  
 "match": true  
}
