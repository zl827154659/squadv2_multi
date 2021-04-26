import concurrent
import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
import threading
from time import sleep

import timeout_decorator
import translators as ts
from google_trans_new import google_translator

# default parameter : url_suffix="cn" timeout=5 proxies={}
from translator.baidu import Baidu
from translator.google import Google

translator = google_translator(url_suffix='cn', timeout=10)
gg = Google()
bd = Baidu()
mutex = threading.Lock()


class Counter:
    def __init__(self):
        self.counter = 0


def translate(source_text: str):
    return translator.translate(source_text.strip(), 'zh')


@timeout_decorator.timeout(10, use_signals=False)
def ts_google_translate(source_text, if_use_cn_host=True, to_language='zh-CN'):
    return ts.google(source_text, if_use_cn_host=if_use_cn_host, to_language=to_language).strip()


def multi_translate(source_text: str, output_file, dataset, counter: Counter, qa):
    result_text = None
    while result_text is None:
        try:
            # 每翻译完100条存一次, 然后counter重置0
            mutex.acquire()
            if counter.counter >= 700:
                counter.counter = 0
                mutex.release()
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f)
                print(f'{threading.currentThread().name}: saved file for counted 100 times :{output_file}')
            else:
                mutex.release()

            # result_text = translate(query_text).strip()
            result_text = gg.translate('en', 'zh-CN', source_text).pop().strip()
        except Exception as e:
            print(f'{threading.currentThread().name}: first google error : {e}, try baidu soon')
            try:
                result_text = bd.translate('en', 'zh', source_text).strip()
            except Exception as e:
                print(f'{threading.currentThread().name}: baidu error again : {e}, try another')
                try:
                    result_text = ts_google_translate(source_text, if_use_cn_host=True, to_language='zh-CN')
                except Exception as e:
                    print(f'{threading.currentThread().name}: error again : {e}, try soon')
                    sleep(10)
    qa['zh_question'] = result_text

    mutex.acquire()
    print(f'translated one paragraph :{result_text}, counter:{counter.counter}')
    counter.counter += 1
    mutex.release()

    return result_text


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


def multi_data_process(file_path, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
    else:
        with open(file_path, 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
    counter = Counter()
    future_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for document in dataset['data']:
            for paragraph in document['paragraphs']:
                for qa in (paragraph['qas']):
                    query_text = qa['question'].strip()
                    try:
                        zh_text = qa['zh_question']
                        continue
                    except KeyError:
                        future = executor.submit(multi_translate, query_text, output_file, dataset, counter, qa)
                        future_list.append(future)
    for future in concurrent.futures.as_completed(future_list):
        pass

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    print(f'final saved file :{output_file}')


if __name__ == "__main__":
    # data_process('./dev.json', './dev_test.json')
    multi_data_process('./train-v2.0.json', './train_zh_question.json')
