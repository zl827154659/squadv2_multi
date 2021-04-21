# 百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import requests
import hashlib
import urllib
import random
import json
import os
from tqdm import *
import re
import time


def get_zh_sentences():
    max_len = 100
    root = "./DATA/"
    filenames = os.listdir(root)
    zh_sentences = []
    for filename in filenames:
        fp = open(root + filename, "r", encoding="utf-8")
        json_data = json.load(fp)
        for data in json_data.values():
            summary = data["summary"]
            content = data["content"]
            if len(summary) > 6:
                zh_sentences.append(summary)
            if len(content) > 6:
                sub_contents = [sent + "。" for sent in content.split("。")]
                zh_sentence = ""
                for sub_content in sub_contents:
                    temp_zh_sentence = zh_sentence + sub_content
                    if len(temp_zh_sentence) > max_len:
                        zh_sentences.append(zh_sentence)
                        zh_sentence = sub_content
                    else:
                        zh_sentence = temp_zh_sentence
    print(len(zh_sentences))
    return zh_sentences


def translate(from_sentence, to_lang):
    appid = '20190921000336211'  # 填写你的appid
    secretKey = '1HVGlYtvJsUIZxpryTzI'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = 'zh'  # 原文语种
    toLang = to_lang  # 译文语种
    salt = random.randint(32768, 65536)
    q = from_sentence
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    return_result = "error"
    try:
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        data = {
            "appid": appid,
            "q": q,
            "from": fromLang,
            "to": toLang,
            "salt": str(salt),
            "sign": sign,
        }
        res = requests.post(url, data=data)
        return_result = json.loads(res.content).get('trans_result')
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()
        return return_result

def main():
    tgt_lang = 'en'
    # line = '只是之前的出尔反尔,估计也是要推出一个重量级人物让他承认是自己未经宗门同意就隐瞒了事实,引咎下台接受门规处置,然后太天门向三家宗门付出一定的补偿,从而了结此事。\n你还想怎样？”花长老虽然火爆,但是却也是知道轻重的一个人,听着杨晨失望的语气,忍不住气的问道。'
    # print(translate(line,tgt_lang))
    src_name = '../data/小说/458本小说分句2290chapters226478sents.json'
    out_path = '../data/小说/458本小说翻译2290chapters226478sents.json'
    first_trans = False

    with open(src_name, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    try:
        if not first_trans:
            with open(out_path, 'r', encoding='utf-8') as out_fp:
                new_data = json.load(out_fp)
            trans_no = list(new_data.items())[-1][0]
            trans_no = int(trans_no) + 1
            print(trans_no)
        else:
            new_data = {}
        pbar = tqdm(total=2290 - trans_no)
        for chapter_no, chapter_content in data.items():
            if int(chapter_no) <= trans_no:
                continue
            i = 0
            en_lack_index = []
            en_results = []
            chapter_pairs = []
            while (i < len(chapter_content)):
                line = ''
                while (len(line) < 1900 and i < len(chapter_content)):
                    pairs = chapter_content[i]
                    chapter_pairs.append(pairs)
                    if pairs[0]:
                        i += 1
                        continue
                    en_lack_index.append(i)
                    line += pairs[1] + '\n'
                    i += 1
                result = translate(line, to_lang=tgt_lang)
                # print(result)
                if (result == None):
                    print('sleeping,retry...')
                    time.sleep(10)
                    main()
                    return
                for j in range(len(result)):
                    en_results.append(result[j]['dst'])
                time.sleep(1)

            for index, lack_index in enumerate(en_lack_index[:-1]):
                try:
                    chapter_pairs[lack_index][0] = en_results[index]
                except:
                    print(chapter_no, index, lack_index)
                    continue

            new_data[chapter_no] = chapter_pairs
            pbar.set_description('trans novels')
            pbar.update(1)
            with open(out_path, 'w', encoding='utf-8') as fp:
                json.dump(new_data, fp, indent=3, ensure_ascii=False)
    finally:
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(new_data, fp, indent=3, ensure_ascii=False)

if __name__ == "__main__":
    # main()
    tgt_lang = 'en'
    # line = '只是之前的出尔反尔,估计也是要推出一个重量级人物让他承认是自己未经宗门同意就隐瞒了事实,引咎下台接受门规处置,然后太天门向三家宗门付出一定的补偿,从而了结此事。\n你还想怎样？”花长老虽然火爆,但是却也是知道轻重的一个人,听着杨晨失望的语气,忍不住气的问道。'
    # print(translate(line,tgt_lang))
    src_name = '../data/小说/458本小说分句1374chapters183587sents.json'
    out_path = '../data/小说/458本小说翻译1374chapters183587sents.json'
    first_trans = False

    with open(src_name, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    try:
        if not first_trans:
            with open(out_path, 'r', encoding='utf-8') as out_fp:
                new_data = json.load(out_fp)
            trans_no = list(new_data.items())[-1][0]
            trans_no = int(trans_no) + 1
            print(trans_no)
        else:
            new_data = {}
            trans_no = 0
        pbar = tqdm(total=1374-trans_no)
        for chapter_no,chapter_content in data.items():
            sleep = False
            if int(chapter_no)<=trans_no:
                continue
            i = 0
            en_lack_index = []
            en_results = []
            chapter_pairs = []
            while(i<len(chapter_content)):
                line = ''
                while(len(line)<1900 and i<len(chapter_content)):
                    pairs = chapter_content[i]
                    chapter_pairs.append(pairs)
                    if pairs[0]:
                        i += 1
                        continue
                    en_lack_index.append(i)
                    line += pairs[1] + '\n'
                    i += 1
                result = translate(line, to_lang=tgt_lang)
                # print(result)
                if(result==None):
                    print('sleeping,retry...')
                    time.sleep(10)
                    # result = translate(line, to_lang=tgt_lang)
                    sleep = True
                    break
                for j in range(len(result)):
                    en_results.append(result[j]['dst'])
                time.sleep(1)

            if sleep: continue

            for index,lack_index in enumerate(en_lack_index[:]):
                try:
                    chapter_pairs[lack_index][0] = en_results[index]
                except:
                    print(chapter_no,index,lack_index)
                    continue

            new_data[chapter_no] = chapter_pairs
            pbar.set_description('trans novels')
            pbar.update(1)
            with open(out_path, 'w', encoding='utf-8') as fp:
                json.dump(new_data, fp, indent=3, ensure_ascii=False)
    finally:
        with open(out_path,'w',encoding='utf-8') as fp:
            json.dump(new_data,fp,indent=3,ensure_ascii=False)