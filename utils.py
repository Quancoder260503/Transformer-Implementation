import torch
import os
import wget
import tarfile
import shutil
import codecs
import youtokentome
import math
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_data(data_folder):
    """
    Downloads the training, validation and test files for VMT ' 14 english-german translation task
    """

    train_url = [
      "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
      "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
      "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
    ]

    if not os.path.isdir(os.path.join(data_folder, "tar_files")):
        os.makedirs(os.path.join(data_folder, "tar_files"))
    # CREATE a fresh folder to extract downloaded TAR files,
    if os.path.isdir(os.path.join(data_folder, "extracted_files")):
        shutil.rmtree(os.path.join(data_folder, "extracted_files"))
        os.makedirs(os.path.join(data_folder, "extracted_files"))

    for url in train_url :
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(data_folder, "tar_files", filename)):
            print(f"\nDownloading {filename}...")
            wget.download(url, os.path.join(data_folder, "tar_files", filename))
        print(f"\nExtracting {filename}...")
        tar = tarfile.open(os.path.join(data_folder, "tar_files", filename))
        members = [m for m in tar.getmembers() if "de-en" in m.path]
        tar.extractall(os.path.join(data_folder, "extracted_files"), members)

    # Download validation and testing data using sacreBLEU
    print("\n")
    os.system("sacrebleu -t wmt13 -l en-de --echo src > '" + os.path.join(data_folder, "val.en") + "'")
    os.system("sacrebleu -t wmt13 -l en-de --echo ref > '" + os.path.join(data_folder, "val.de") + "'")
    print("\n")
    os.system("sacrebleu -t wmt14 -l en-de --echo src > '" + os.path.join(data_folder, "test.en") + "'")
    os.system("sacrebleu -t wmt14 -l en-de --echo ref > '" + os.path.join(data_folder, "test.de") + "'")

    dirs = [d for d in os.listdir(os.path.join(data_folder, "extracted_files"))
            if os.path.isdir(os.path.join(data_folder, "extracted_files", d))]
    for dir in dirs :
        for f in os.listdir(os.path.join(data_folder, "extracted_files", dir)):
            shutil.move(os.path.join(data_folder, "extracted_files", dir, f),
                        os.path.join(data_folder, "extracted_files"))
        os.remove(os.path.join(data_folder, "extracted_files", dir))

def prepare_data(data_folder, euro_parl = False, common_crawl = True, new_commentary = True,
                 min_length = 3, max_length = 100, max_length_ratio = 1.5, retain_case = True):
  """
  Filters and prepares the training data
  :param data_folder: the folder where the files were stored
  :param euro_parl:
  :param common_crawl:
  :param new_commentary:
  :param min_length: exclude sequence pair where one or both are shorter than the min BPE length
  :param max_length: exclude sequence pair where one or both are longer than the max BPE length
  :param max_length_ratio: exclude sequence pair where one are much longer than the other
  :param retain_case: retain the case or not ?
  :return:
  """

  german  = list()
  english = list()
  files   = list()
  assert euro_parl or common_crawl or new_commentary, "Set at least one of these to be true"

  if euro_parl :
      files.append("europarl-v7.de-en")
  if common_crawl :
      files.append("commoncrawl.de-en")
  if new_commentary :
      files.append("news-commentary-v9.de-en")

  print(f"\n Reading extracted filed and combining ...")

  for file in files :
      with codecs.open(os.path.join(data_folder, "extracted_files", file + ".de"), "r", encoding = "utf-8") as f:
          if retain_case :
              german.extend(f.read().split("\n"))
          else:
              german.extend(f.read().lower().split("\n"))

      with codecs.open(os.path.join(data_folder, "extracted_files", file + ".en"), "r", encoding = "utf-8") as f :
          if retain_case :
              english.extend(f.read().split("\n"))
          else :
              english.extend(f.read().lower().split("\n"))
      assert len(english) == len(german)

  print("\nWriting to a single files...")
  with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding = "utf-8") as f :
      f.write("\n".join(english))
  with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding = "utf-8") as f :
      f.write("\n".join(german))
  with codecs.open(os.path.join(data_folder, "train.en-de"), "w", encoding = "utf-8") as f:
      f.write("\n".join(english + german))

  del english, german

  print("\nLearning BPE...")
  youtokentome.BPE.train(data = os.path.join(data_folder, "train.en-de"), vocab_size = 37000,
                         model = os.path.join(data_folder, "bpe.model"))
  print("\nRe-reading BPE model...")
  bpe_model = youtokentome.BPE(os.path.join(data_folder, "bpe.model"))

  print("\nRe-reading single-files...")
  with codecs.open(os.path.join(data_folder, "train.en"), "r", encoding = "utf-8") as f :
      english = f.read().split("\n")
  with codecs.open(os.path.join(data_folder, "train.de"), "r", encoding = "utf-8") as f :
      german  = f.read().split("\n")

  print("\nFiltering...")

  pairs = list()
  for en, de in tqdm(zip(english, german), total = len(english)):
      en_tok = bpe_model.encode(en, output_type = youtokentome.OutputType.ID)
      de_tok = bpe_model.encode(de, output_type = youtokentome.OutputType.ID)
      len_en_tok = len(en_tok)
      len_de_tok = len(de_tok)

      if min_length < len_en_tok < max_length and \
         min_length < len_de_tok < max_length and \
         1. / max_length_ratio <= len_en_tok / len_de_tok <= max_length_ratio :
           pairs.append((en, de))
      else :
          continue

  english, german = zip(*pairs)
  print("\nRe-writing filtered sentences to single files...")
  os.remove(os.path.join(data_folder, "train.en"))
  os.remove(os.path.join(data_folder, "train.de"))
  os.remove(os.path.join(data_folder, "train.en-de"))

  with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding = "utf-8") as f :
      f.write("\n".join(english))
  with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding = "utf-8") as f :
      f.write("\n".join(german))

  del  english, german, bpe_model, pairs

  print("\nFINISH...\n")

def get_positional_encoding(d_model, max_length = 100):
    """
    Computes the positional encoding
    :param d_model: size of the vectors through the transformer model
    :param max_length: maximum length of a sequence up to which a positional encoding can be computed
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    positional_encoding = torch.zeros((max_length, d_model))
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else :
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))
    positional_encoding = positional_encoding.unsqueeze(0) # (1, max_length, d_model)
    return positional_encoding

def get_learning_rate(step, d_model, warmup_steps):
    """
    The LR scheduler
    :param step: training step number
    :param d_model: size of vectors through the transformer model
    :param warmup_steps: number of warmup steps before the learning rate is increasing linearly
    :return: updated learning rate
    """
    lr = 2 * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    return lr

def save_checkpoint(epoch, model, optimizer, prefix_dir_name):
    """
    Checkpoint saver. Each save overwrites the previous one
    :param epoch: epoch number
    :param model: transformer model weight
    :param optimizer: optimizer state
    :param prefix_dir_name: check point filename prefix
    """
    state = {
        'epoch' : epoch,
        'model' : model,
        'optimizer' : optimizer
    }
    filename = prefix_dir_name + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)

def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor
    :param optimizer: optimizer whose learning rate will be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr



class AverageTracker(object):
    """
    Keep tracks of the most recent, average, sum and count of metrics
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count