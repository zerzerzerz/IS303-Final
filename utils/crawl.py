import json
from selenium import webdriver
import requests
from lxml import etree
from utils.utils import load_json, save_json


def get_car_type():
    tmp = '/html/body/div[3]/div[1]/div/div[1]/div[2]/div/div/p/span[2]/a[@href]'
    driver = webdriver.Chrome()
    ans = []
    driver.get('https://www.renrenche.com/cn/swmsiweiqiche/')
    for a in driver.find_elements_by_xpath(tmp):
        # print()
        ans.append(a.get_attribute('href'))
    print(len(ans))
    save_json(ans,'data/car_type2.json')


def get_car_id(num_page=1):
    driver = webdriver.Chrome()
    driver.get('https://www.bilibili.com')
    ans = []
    count = 0
    for type_ in load_json('data/car_type2.json'):
        for page in (range(1,num_page+1)):
            driver.get(f'{type_}p{page}/?')
            car_ul = driver.find_element_by_xpath('//*[@id="search_list_wrapper"]/div/div/div[1]/ul')
            li_s = car_ul.find_elements_by_xpath('./li')
            # print(len(li_s))
            for li in li_s:
                a = li.find_element_by_xpath('./a')
                # print(a)
                car_id = a.get_attribute('data-car-id')
                # print(car_id)
                ans.append(car_id)
                count += 1
                print(count)
                # //*[@id="list_item_href/3debbaae558482fb"]
                # /html/body/div[3]/div[3]/div/div/div[1]/ul/li[1]/a
    save_json(ans,'data/car_id3.json')


def get_car_info(car_id,driver):
    try:
        driver.get(f'https://www.renrenche.com/bj/car/{car_id}')
        ans = {}
        ans['title'] = driver.find_element_by_xpath('/html/body/div[5]/div[1]/div[2]/div[2]/div[1]/div[1]/h1').text
        ans['price'] = driver.find_element_by_xpath('//p[@class="price detail-title-right-tagP"]').text
        ul = driver.find_element_by_xpath('//ul[@class="row-fluid list-unstyled box-list-primary-detail"]')
        ans['kilometer'] =       ul.find_element_by_xpath('./li[1]/div/p[1]/strong').text
        ans['shangpai-date'] =   ul.find_element_by_xpath('./li[2]/div/p[1]/strong').text
        ans['chepai-location'] = ul.find_element_by_xpath('./li[3]/div/p[1]/strong').text
        ans['standard'] =        ul.find_element_by_xpath('./li[4]/div/p[1]/strong').text
        ans['change-speed'] =    ul.find_element_by_xpath('./li[5]/div/p[1]/strong').text
        # /html/body/div[5]/div[1]/div[2]/div[2]/div[1]/div[5]/ul/li[6]/p[1]/strong
        # print(ans)
        return ans
    except:
        {}


def get_car_info2(car_id,*args,**kwds):
    try:
        res = requests.get(f'https://www.renrenche.com/cn/car/{car_id}')
        tree = etree.HTML(res.text)
        ans = {}
        ans['title'] = tree.xpath('/html/body/div[5]/div[1]/div[2]/div[2]/div[1]/div[1]/h1/text()')[1].strip()
        ans['price'] = tree.xpath('//p[@class="price detail-title-right-tagP"]/text()')[0]
        ans['kilometer'] = tree.xpath('//ul[@class="row-fluid list-unstyled box-list-primary-detail"]/li[1]/div/p[1]/strong/text()')[0]
        ans['shangpai-date'] = tree.xpath('//ul[@class="row-fluid list-unstyled box-list-primary-detail"]/li[2]/div/p[1]/strong/text()')[0]
        ans['chepai-location'] = tree.xpath('//ul[@class="row-fluid list-unstyled box-list-primary-detail"]/li[3]/div/p[1]/strong/text()')[0]
        ans['standard'] = tree.xpath('//ul[@class="row-fluid list-unstyled box-list-primary-detail"]/li[4]/div/p[1]/strong/text()')[0]
        ans['change-speed'] = tree.xpath('//ul[@class="row-fluid list-unstyled box-list-primary-detail"]/li[5]/div/p[1]/strong/text()')[0]
        return ans
    except:
        return {}
