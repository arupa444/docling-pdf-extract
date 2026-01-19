from scrapy.crawler import CrawlerProcess
from utils.spider import FullWebsiteSpider
import sys

target_url = sys.argv[1]

process = CrawlerProcess({
    "LOG_LEVEL": "INFO",
})

process.crawl(
    FullWebsiteSpider,
    start_url=target_url
)

process.start()