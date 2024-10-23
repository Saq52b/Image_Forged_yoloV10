import os
import pyodbc
import requests
import uuid
conn_str = (
    r"Driver={SQL Server};"
    r"Server=192.168.0.152,151;"  
    r"Database=AML_SHOP_CANADA_2024;"  
    r"Trusted_Connection=yes;"  
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    print("Connected to the database!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()
query = "SELECT ROW_ID_GID,ImgIdFrontUrl FROM tbl_MobApp_Customer where ImgIdFrontUrl <> ''"  

try:
    cursor.execute(query)
    rows = cursor.fetchall()
    if not os.path.exists('img'):
        os.makedirs('img')
    for row in rows:
        customer_id = row[0]
        image_url = row[1] 
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                img_file_path = f'img/customer_{customer_id}.jpg'
                with open(img_file_path, 'wb') as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
                
                print(f"Image for customer {customer_id} saved as {img_file_path}")
            else:
                print(f"Failed to download image for customer {customer_id}, URL: {image_url}")
        except Exception as e:
            print(f"Error downloading image for customer {customer_id}: {e}")

except Exception as e:
    print(f"Error fetching or processing data: {e}")
finally:
    cursor.close()
    conn.close()
