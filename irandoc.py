import scrapy
import json
import urllib.request

class IrandocSpider(scrapy.Spider):
    name = 'irandoc'
    start_urls = ['https://ganj.irandoc.ac.ir/api/v1/search/main?basicscope=1&fulltext_status=1&keywords=%D8%B9%D9%84%DB%8C+%D8%B1%D8%AC%D8%A7%D8%A6%DB%8C&limitation=organization_9857,organization_451&results_per_page=4&sort_by=1&year_from=0&year_to=1400%2Fshow_tags']

    def parse(self, response):
        
        data = json.loads(response.text)
        print(response.text)
        
        for d in data['results']:
            
             req = urllib.request.Request('https://ganj.irandoc.ac.ir/api/v1/articles/'+d['uuid']+'/show_tags')
             with urllib.request.urlopen(req) as response:
                 html = response.read()
                 
             keywords = json.loads(html)
             
        for k in keywords['tags']:
             yield { 'tags':k,
                     'key':html
                        }
        return {'Tag':keywords['tags']}
          
        with open('keywords_f.json', "w") as f:
            f.writelines(keywords['tags'])
        f.close('keywords_f.json')