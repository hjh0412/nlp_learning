import requests
import uuid
import time
import hashlib
appKey = "60dd45f0145c18c4"
appSecret = "d13ZglkDh5ztlaUfCHLuM67r7CmZeLsQ"
url = "https://openapi.youdao.com/api"


def truncate(text):
    if len(text) <= 20:
        return text
    else:
        return text[:10] + str(len(text)) + text[-10:]


def translate(text,from_lang="zh-CHS",to_lang="en"):
    data = {"from": "zh-CHS", "to": "en", "i": text, "doctype": "json",
            "appKey": "",}

    salt = str(uuid.uuid4())          # 随机串
    curtime = str(int(time.time()))   # 秒级时间戳
    # v3签名公式 sha256(appKey + input + salt + curtime + appSecret)
    sign_str = appKey + truncate(text) + salt + curtime + appSecret
    sign = hashlib.sha256(sign_str.encode("utf-8")).hexdigest()

    data = {"from": from_lang, "to": to_lang, "q": text, "doctype": "json",
            "appKey": appKey,
            "sign": sign,
            "signType": "v3",
            "curtime": curtime,
            "salt": salt,}


    result = requests.post(url, data=data,headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"}).json()

    # print("API 返回内容:", result)
    return result['translation'][0]
if __name__=='__main__':
    # print(translate("今天天气很好"))
    print(translate(translate("今天天气不错"),from_lang='en',to_lang='zh-CHS'))    