import requests

url = "https://fall.cv-gml.ru/task_file/15/additional_files"

headers = {
    "Cookie": "session=.eJwlzs1KBTEMQOF36dpFkuZnel9mSNMERVGY0ZXcd7fg9oMD57eddeX92h7lH3e-tPNttUfj5RC1hJU1M7APGUM3iodCmU9FDF-SLOQ6vc9pgIcw9-LOHANKBFkHroqQg0GqOKZhArGqrCQjWEyTUxAiJdC4wKIGtz3yc-f1f0PEY0vcV53fX-_5uc3Ra6ZBHgbeYbnKri2dRAgFyg-VgmzPP9v1QBE.Z4Tjug.mhYU5xb79OKI-ezR_Icya5Jght0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("additional_files.zip", "wb") as f:
        f.write(response.content)
    print("Файл успешно скачан и сохранен как 'additional_files.zip'.")
else:
    print(f"Ошибка при скачивании. Код ответа: {response.status_code}")
    print(f"Тело ответа:\n{response.text}")
import zipfile

zip_file_path = "additional_files.zip"  

output_dir = "/content/additional_files"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Файлы распакованы в {output_dir}")





