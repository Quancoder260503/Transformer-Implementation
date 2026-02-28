from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(BASE_DIR, 'database/transformer_data')

download_data(data_folder = data_folder)

prepare_data(data_folder = data_folder,
             euro_parl = True,
             common_crawl = True,
             new_commentary = True,
             min_length = 3,
             max_length = 150,
             max_length_ratio = 2.0,
             retain_case = True)






