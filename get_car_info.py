from utils.crawl import get_car_info, get_car_info2
from utils.utils import load_json
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# driver = webdriver.Chrome()

ids = load_json('data/car_id3.json')
num_id = len(ids)
finished_df_index = -1

block_size = 100
for df_index in tqdm(range(0,num_id,block_size)):
    if df_index < finished_df_index:
        continue
    ans = pd.DataFrame()
    for car_id in ids[df_index:df_index+block_size]:
        if car_id is None:
            continue
        info = get_car_info2(car_id)
        ans = ans.append(info,ignore_index=True)
    ans.to_csv('data/car_info_requests_%06d.csv' % (df_index // block_size), index=False)
# print(info)
