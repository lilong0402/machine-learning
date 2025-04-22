
import os
import pandas as pd


class MyDataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

class MyDataProcessor(DataProcessor):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    file_path = os.path.join(data_dir, 'train_sentiment.txt')
    f = open(file_path, 'r', encoding='utf-8')  # 读取数据，并指定中文常用的utf-8
    train_data = []
    index = 0  # ID值
    for line in f.readlines():  # 参考XnliProcessor
        guid = "train-%d" % index
        line = line.replace('\n', '').split('\t')  # 处理换行符，原数据是以tab分割
        text_a = tokenization.convert_to_unicode(str(line[1]))  # 第0位置是索引，第1位置才是数据，可以查看train_sentiment.txt
        label = str(line[2])  # 我们的label里没有什么东西，只有数值，所以转字符串即可
        train_data.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))  # 这里我们没text_b，所以传入None
        index += 1  # index每次不一样，所以加等于1
    return train_data  # 这样数据就读取完成

class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)  # 获取样本ID
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])  # 获取text_a和b，我们只有a所以把b去掉
      label = tokenization.convert_to_unicode(line[2])  # 获取标签
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))  # 把读进来的东西传到InputExample，这个类可以点进去，里面什么都没做，只不过是模板，我们也照着做
    return examples



if __name__ == '__main__':
    path = "my_model_predict"
    pd_all = pd.read_csv(os.path.join(path, "test_results.tsv"), sep='\t', header=None)

    data = pd.DataFrame(columns=['polarity'])
    print(pd_all.shape)

    for index in pd_all.index:
        neutral_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]
        negative_score = pd_all.loc[index].values[2]

        if max(neutral_score, positive_score, negative_score) == neutral_score:
            data.loc[index+1] = ["0"]
        elif max(neutral_score, positive_score, negative_score) == positive_score:
            data.loc[index+1] = ["1"]
        else:
            data.loc[index+1] = ["2"]

    data.to_csv(os.path.join(path, "pre_sample.tsv"), sep='\t')