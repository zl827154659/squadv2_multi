import json
import os
from time import sleep
import translators as ts
from google_trans_new import google_translator

# default parameter : url_suffix="cn" timeout=5 proxies={}
from translator.baidu import Baidu
from translator.google import Google

translator = google_translator(url_suffix='cn', timeout=10)
gg = Google()
bd = Baidu()


def translate(source_text: str):
    return translator.translate(source_text.strip(), 'zh')


def data_process(file_path, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
    else:
        with open(file_path, 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
    counter = 0
    for document in dataset['data']:
        for paragraph in document['paragraphs']:
            for qa in (paragraph['qas']):
                query_text = qa['question'].strip()
                try:
                    zh_text = qa['zh_question']
                    continue
                except KeyError:
                    zh_text = None
                    while zh_text is None:
                        try:
                            # 每翻译完100条存一次, 然后counter重置0
                            if counter >= 100:
                                with open(output_file, 'w', encoding='utf-8') as f:
                                    json.dump(dataset, f)
                                    print(f'saved file for counted 100 times :{output_file}')
                                counter = 0
                            # zh_text = translate(query_text).strip()
                            zh_text = gg.translate('en', 'zh-CN', query_text).pop().strip()
                        except Exception as e:
                            # 出现报错后，把counter重置0，因为接下来马上要保存了
                            print(f'gg error occurred!:{e}, will try another')
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(dataset, f)
                                print(f'saved file for error :{output_file}')
                            counter = 0
                            try:
                                zh_text = bd.translate('en', 'zh', query_text).pop().strip()
                            except Exception as e:
                                print(f'baidu error again : {e}, try another')
                                try:
                                    zh_text = ts.google(query_text, if_use_cn_host=True, to_language='zh-CN',
                                                        sleep_seconds=1).strip()
                                except Exception as e:
                                    print(f'error again : {e}, try soon')
                                    sleep(10)
                    qa['zh_question'] = zh_text
                    print(f'translated one paragraph :{zh_text}, counter:{counter}')
                    counter += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
        print(f'final saved file :{output_file}')


if __name__ == "__main__":
    data_process('./dev-v2.0.json', './dev_zh_question.json')
