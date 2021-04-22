一次跨语言开放式检索问答的实验

语言模型：bert-base-multilingual-cased

数据集来源：squad V2（其中所有的question都被翻译成了中文问题，在zh-question字段）
            read_examples时按zh-question字段读取，观察模型的跨语言性能

实验结果：
    未知