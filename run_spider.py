from scrapy.crawler import CrawlerProcess
from utils.spider import FullWebsiteSpider

process = CrawlerProcess({
    "LOG_LEVEL": "INFO",
})

process.crawl(
    FullWebsiteSpider,
    start_url="https://arupas-portfolio.vercel.app/"
)

process.start()