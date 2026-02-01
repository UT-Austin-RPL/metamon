import os
import re
import argparse
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://www.smogon.com/stats/"

parser = argparse.ArgumentParser(
    description="Gathers tier usage statistics from Smogon by month across a range of years"
)
parser.add_argument(
    "--start_date",
    type=int,
    default=2015,
    help="Start date for scraping (YYYY) (inclusive)",
)
parser.add_argument(
    "--end_date",
    type=int,
    default=2024,
    help="End year for scraping (YYYY) (exclusive)",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./stats",
    help="Local directory to save the scraped files",
)
parser.add_argument(
    "--baselines",
    type=str,
    default="",
    help=(
        "Comma-separated baselines to keep (e.g. '0,1500,1695,1825'). "
        "Empty = keep all."
    ),
)
parser.add_argument(
    "--min_baseline",
    type=float,
    default=None,
    help="If set, only download files with baseline >= this value.",
)
parser.add_argument(
    "--include_chaos",
    action="store_true",
    help="Download chaos/ JSON files (includes info.cutoff metadata).",
)
parser.add_argument(
    "--max_concurrency",
    type=int,
    default=8,
    help="Maximum number of concurrent HTTP requests.",
)
parser.add_argument(
    "--max_retries",
    type=int,
    default=5,
    help="Maximum number of retries for a failed request.",
)
parser.add_argument(
    "--backoff_base",
    type=float,
    default=0.5,
    help="Base seconds for exponential backoff (retry delay = base * 2^attempt).",
)
args = parser.parse_args()


BASELINE_RE = re.compile(r"-(\d+(?:\.\d+)?)\.(txt|json)$")

SKIP_DIRS = {"monotype", "metagame", "leads"}
if not args.include_chaos:
    SKIP_DIRS.add("chaos")

allowed_baselines = None
if args.baselines.strip():
    allowed_baselines = {
        float(x.strip()) for x in args.baselines.split(",") if x.strip()
    }


def extract_baseline(href: str):
    m = BASELINE_RE.search(href)
    if m:
        return float(m.group(1))
    if href.endswith(".txt") or href.endswith(".json"):
        # Smogon convention: no explicit baseline means 1500.
        return 1500.0
    return None


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


async def save_text_file(session, url, local_path):
    # Check if the file already exists
    if os.path.isfile(local_path):
        print(f"File already exists: {local_path}")
        return
    async with session.get(url) as response:
        if response.status == 200:
            text = await response.text()
            async with aiofiles.open(local_path, "w", encoding="utf-8") as file:
                await file.write(text)


async def _fetch_with_retries(session, url):
    for attempt in range(args.max_retries + 1):
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                return await response.text()
        except Exception:
            if attempt >= args.max_retries:
                raise
            await asyncio.sleep(args.backoff_base * (2**attempt))


async def scrape_base(session, url, local_dir, start_date, end_date, sem):
    async with sem:
        text = await _fetch_with_retries(session, url)
    soup = BeautifulSoup(text, "html.parser")
    tasks = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and not href.startswith("?") and href != "../":
            href_date = int(href[:4])
            href_full = urljoin(url, href)
            local_path = os.path.join(local_dir, href)

            if (
                href.endswith("/") and href_date >= start_date and href_date < end_date
            ):  # It's a directory
                ensure_dir(local_path)
                task = asyncio.create_task(scrape(session, href_full, local_path, sem))
                tasks.append(task)

    await asyncio.gather(*tasks)


async def scrape(session, url, local_dir, sem):
    try:
        async with sem:
            text = await _fetch_with_retries(session, url)
        soup = BeautifulSoup(text, "html.parser")

        tasks = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if not href or href.startswith("?"):
                continue
            if href.endswith("/"):
                if href == "../":
                    continue
                dirname = href.rstrip("/")
                if dirname in SKIP_DIRS:
                    continue
                href_full = urljoin(url, href)
                local_path = os.path.join(local_dir, href)
                ensure_dir(local_path)
                task = asyncio.create_task(scrape(session, href_full, local_path, sem))
                tasks.append(task)
                continue

            if href.endswith(".txt") or href.endswith(".json"):
                baseline = extract_baseline(href)
                if (
                    allowed_baselines is not None
                    and baseline is not None
                    and baseline not in allowed_baselines
                ):
                    continue
                if (
                    args.min_baseline is not None
                    and baseline is not None
                    and baseline < args.min_baseline
                ):
                    continue
                href_full = urljoin(url, href)
                local_path = os.path.join(local_dir, href)
                print(f"Downloading {href_full} to {local_path}")
                task = asyncio.create_task(
                    save_text_file(session, href_full, local_path)
                )
                tasks.append(task)

        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error on url {url}: {e}")


ensure_dir(args.save_dir)


async def main():
    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    sem = asyncio.Semaphore(args.max_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await scrape_base(
            session,
            base_url,
            args.save_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            sem=sem,
        )


asyncio.run(main())
