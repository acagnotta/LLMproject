"""Polite web crawler: collects pages from whole websites using their sitemaps.

The list of websites lives in config/corpora/<corpus>.yaml (key `sites`).

A sitemap is an XML file where a website lists its own pages. Reading it is
faster and more respectful than following every link on every page.

Rules we follow:
  - skip URLs disallowed by the site's robots.txt
  - wait `delay_seconds` between requests (config/settings.yaml)
  - keep only URLs matching `include` and stop after `max_pages` per site

Run with:  python src/crawl.py --corpus fantacalcio
"""
import json
import re
import time
from datetime import date, timedelta
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from config import CORPORA, SETTINGS, corpus_paths, get_corpus

USER_AGENT = SETTINGS["crawler"]["user_agent"]
DELAY = SETTINGS["crawler"]["delay_seconds"]

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def get(url: str) -> str:
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def expand_sitemaps(site: dict) -> list[str]:
    """Replace {date:...} placeholders with each of the last `days_back` days."""
    today = date.today()
    days = range(site.get("days_back", 1))
    result = []
    for sm in site["sitemaps"]:
        if "{date" in sm:
            result += [sm.format(date=today - timedelta(d)) for d in days]
        else:
            result.append(sm)
    return result


def sitemap_urls(sitemap_url: str) -> list[str]:
    """Return page URLs from a sitemap (follows nested sitemap indexes)."""
    try:
        xml = get(sitemap_url)
    except requests.RequestException as e:
        print(f"  ! sitemap unavailable: {sitemap_url} ({e})")
        return []
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml)
    if "<sitemapindex" in xml:   # an index of other sitemaps
        return [u for sub in locs for u in sitemap_urls(sub)]
    return locs


def html_to_text(html: str) -> tuple[str, str]:
    """Extract (title, main readable text) from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return title, main.get_text(separator=" ")


def keep_latest(urls: list[str], n: int) -> list[str]:
    """For numbered pages like .../ultime-dai-campi/11, keep only the n highest numbers
    of each section (= the most recent matchdays). Other URLs are kept as they are."""
    sections: dict[str, list[int]] = {}
    for u in urls:
        if m := re.match(r"(.*)/(\d+)/?$", u):
            sections.setdefault(m[1], []).append(int(m[2]))
    latest = {f"{prefix}/{num}" for prefix, nums in sections.items()
              for num in sorted(nums)[-n:]}
    return [u for u in urls if not re.match(r".*/\d+/?$", u) or u.rstrip("/") in latest]


def crawl_site(site: dict, out) -> int:
    print(f"\n=== {site['name']}")
    robots = RobotFileParser(site["robots"])
    robots.read()

    # 1. Collect candidate URLs from the sitemaps (deduplicated, order preserved)
    urls = list(dict.fromkeys(u for sm in expand_sitemaps(site) for u in sitemap_urls(sm)))
    urls = [u for u in urls if re.search(site["include"], u) and robots.can_fetch(USER_AGENT, u)]
    if "keep_latest" in site:   # drop old matchdays: their info is outdated
        urls = keep_latest(urls, site["keep_latest"])
    urls = urls[: site["max_pages"]]
    print(f"{len(urls)} pages to download (~{len(urls) * DELAY / 60:.0f} min)")

    # 2. Download each page and save its text
    saved = 0
    for i, url in enumerate(urls, 1):
        try:
            title, text = html_to_text(get(url))
        except requests.RequestException as e:
            print(f"  [{i}/{len(urls)}] SKIPPED {url}: {e}")
            continue
        record = {"url": url, "site": site["name"], "title": title,
                  "text": text, "crawled_at": date.today().isoformat()}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        saved += 1
        print(f"  [{i}/{len(urls)}] {url}")
        time.sleep(DELAY)
    return saved


def main() -> None:
    corpus = get_corpus()
    sites = CORPORA[corpus].get("sites") or []
    if not sites:
        print(f"No sites configured for corpus '{corpus}' (see config/corpora/{corpus}.yaml).")
        return
    out_file = corpus_paths(corpus)["crawled"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as out:
        total = sum(crawl_site(site, out) for site in sites)
    print(f"\nSaved {total} pages to {out_file}")


if __name__ == "__main__":
    main()
