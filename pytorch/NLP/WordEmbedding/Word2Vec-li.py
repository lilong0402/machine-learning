
import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 准备数据与预处理
def data_process(pathroot:(str)):

    # 定义输入输出
    basename = pathroot
    inp = basename + 'zhwiki-latest-pages-articles.xml.bz2'
    outp = basename + 'wiki.zh.text'

    program = os.path.basename(basename)
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    space = " "
    i = 0
    output = open(outp, 'w', encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
    return wiki

# 训练数据
def train():
    # 定义输入输出
    basename = "F:/temp/DL/"
    inp = basename + 'wiki.zh.text'
    outp1 = basename + 'wiki.zh.text.model'
    outp2 = basename + 'wiki.zh.text.vector'

    program = os.path.basename(basename)
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())

    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
train()

