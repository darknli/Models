import codecs
import jieba
import numpy as np
from keras.utils import to_categorical

def participles(src, dest):
    lines = []
    with open(src) as f:
        for line in f:
            lines.append(jieba.cut(line.strip())+['\n'])

    with open(dest, 'w') as f:
        f.writelines(lines)
    print('jieba done')


class Data():
    def __init__(self, data_path):
        self.classes = self.get_vocab(data_path)

    def get_vocab(self, data_path):
        classes = ['bos', 'eos']
        with codecs.open(data_path, encoding='utf-8') as f:
            for line in f:
                for word in line.split(' '):
                    if word not in classes:
                        classes.append(word)
        print('got vocabulary')
        return classes

    def get_data(self, path, bptt=30):
        data = []
        label = []
        count = 0
        bptt += 1
        vocab_size = self.get_vocab_size()
        with codecs.open(path, encoding='utf-8') as f:
            for line in f:
                words = line.split(' ')
                words = [self.classes.index(word) for word in words]
                padding = bptt - len(words)
                if padding > 0:
                    words += [self.classes.index('eos')]*padding
                else:
                    words = words[:bptt]
                data.append(words[:-1])
                label.append(to_categorical(words[1:], vocab_size).reshape(-1))
                # label.append(np.array(words[1:]).reshape(bptt-1, 1))
                count += 1
        return np.array(data), np.array(label)

    def get_vocab_size(self):
        return len(self.classes)


def split_para(src, dest, span=20):
    write_line = []
    with codecs.open(src, encoding='utf-8') as f:
        for lines in f:
            words = lines.split(' ')
            end = span
            while end < len(words):
                begin = end
                end = min(end+span, len(words))
                write_line.append(' '.join(words[begin:end]) + '\n')

    with codecs.open(dest, encoding='utf-8', mode='w') as f:
        f.writelines(write_line)

