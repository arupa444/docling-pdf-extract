import scrapy
import os
from urllib.parse import urlparse, urljoin


class FullWebsiteSpider(scrapy.Spider):
    name = "full_website"

    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_url = start_url
        self.allowed_domain = urlparse(start_url).netloc
        self.start_urls = [start_url]

        # =====================
        # DYNAMIC DIRECTORY SETUP
        # =====================
        # 1. Define the base path: data/domain.com/
        self.base_dir = os.path.join("storeCurlData", self.allowed_domain)

        # 2. Define sub-paths: data/domain.com/html and data/domain.com/files
        self.html_dir = os.path.join(self.base_dir, "html")
        self.files_dir = os.path.join(self.base_dir, "files")

        # 3. Create the directories
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)

    def parse(self, response):
        url = response.url
        content_type = response.headers.get("Content-Type", b"").decode()

        # Sanitize filename
        safe_name = (
            url.replace("https://", "")
            .replace("http://", "")
            .replace("/", "_")
            .replace("?", "_")
        )

        # =====================
        # HTML PAGE
        # =====================
        if "text/html" in content_type:
            # Save inside the specific domain's HTML folder
            path = os.path.join(self.html_dir, f"{safe_name}.html")

            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)

            self.logger.info(f"HTML saved: {path}")

            # ✅ extract links ONLY from HTML
            links = response.css("a::attr(href)").getall()

            for link in links:
                absolute = urljoin(url, link)
                parsed = urlparse(absolute)

                if parsed.netloc == self.allowed_domain:
                    yield scrapy.Request(
                        absolute,
                        callback=self.parse,
                        dont_filter=False
                    )

        # =====================
        # NON-HTML FILE (PDF, IMG, etc.)
        # =====================
        else:
            ext = content_type.split("/")[-1].split(";")[0]

            # Save inside the specific domain's FILES folder
            path = os.path.join(self.files_dir, f"{safe_name}.{ext}")

            with open(path, "wb") as f:
                f.write(response.body)

            self.logger.info(f"FILE saved: {path}")

            # ❌ DO NOT extract links here
            return
