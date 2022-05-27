import requests
from lxml import etree

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