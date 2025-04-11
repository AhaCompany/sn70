import os
import time
import argparse
import traceback
import bittensor as bt
import json
import random
import re
import urllib.parse
from typing import Tuple, List, Dict, Set
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from dotenv import load_dotenv

from shared.log_data import LoggerType
from shared.proxy_log_handler import register_proxy_log_handler
from shared.veridex_protocol import VericoreSynapse, SourceEvidence

# debug
bt.logging.set_trace()

load_dotenv()

# Trusted and diverse domains for high-quality sources
TRUSTED_DOMAINS = [
    "wikipedia.org", "britannica.com", "nature.com", "sciencemag.org", 
    "nasa.gov", "cdc.gov", "nih.gov", "who.int", "un.org", "europa.eu",
    "nationalgeographic.com", "smithsonianmag.com", "bbc.com", "reuters.com",
    "apnews.com", "pnas.org", "scientificamerican.com", "mayoclinic.org", 
    "harvard.edu", "stanford.edu", "mit.edu", "berkeley.edu", "ox.ac.uk",
    "cam.ac.uk", "nytimes.com", "wsj.com", "economist.com"
]

# Keyword list for nonsense detection
NONSENSE_KEYWORDS = [
    "unicorn", "dragon", "fairy", "elves", "magic", "spell", "wizard", "witch", "sorcerer",
    "impossible", "never happened", "fictional", "fantasy", "mythical", "nonexistent",
    "flat earth", "time travel", "invisibility", "teleportation", "levitation", "psychic"
]

class CustomMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_bittensor_objects()
        self.setup_logging()
        
        # Used to track and avoid domain reuse within a single request
        self.current_domains: Set[str] = set()
        
        # Create a throttled client session pool
        self.setup_web_client()
        
        # Initialize search API credentials
        self.setup_search_apis()

    def setup_search_apis(self):
        """Set up multiple search APIs for diversity and redundancy"""
        # Primary API keys (add your actual API keys in .env file)
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.google_cse_id = os.environ.get("GOOGLE_CSE_ID", "")
        self.bing_api_key = os.environ.get("BING_API_KEY", "")
        self.perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        
        # Set up OpenAI client for Perplexity if available
        if self.perplexity_api_key:
            from openai import OpenAI
            self.perplexity_client = OpenAI(
                api_key=self.perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
        else:
            self.perplexity_client = None
            bt.logging.warning("No PERPLEXITY_API_KEY found. Will use alternative search methods.")

    def setup_web_client(self):
        """Set up async HTTP client with connection pooling"""
        # This will be initialized in run() to avoid event loop issues
        self.session = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
    
    async def create_aiohttp_session(self):
        """Create an aiohttp ClientSession for async requests"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            connector = aiohttp.TCPConnector(limit=20, ssl=False)  # Connection pooling
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.headers
            )
        return self.session

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--custom", default="my_custom_value", help="Adds a custom value.")
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID.")
        parser.add_argument("--max_snippets", type=int, default=5, help="Maximum snippets to return per request")
        parser.add_argument("--max_search_results", type=int, default=20, help="Maximum search results to fetch")
        parser.add_argument("--async_timeout", type=int, default=25, help="Async timeout in seconds")
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)

        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "custom_miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running custom miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        bt.logging.info(self.config)

    def setup_proxy_logger(self):
        bt_logger = logging.getLogger("bittensor")
        register_proxy_log_handler(bt_logger, LoggerType.Miner, self.wallet)

    def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered.\nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner on uid: {self.my_subnet_uid}")

    def blacklist_fn(self, synapse: VericoreSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, None
        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, None

    def veridex_forward(self, synapse: VericoreSynapse) -> VericoreSynapse:
        """
        Main handler for incoming validation requests.
        """
        start_time = time.time()
        request_id = synapse.request_id or str(int(time.time()))
        bt.logging.info(f"{request_id} | Received Veridex request")
        
        # Reset domain tracking for this request
        self.current_domains = set()
        
        # Check if statement is likely nonsense
        statement = synapse.statement
        if self.is_likely_nonsense(statement):
            bt.logging.info(f"{request_id} | Statement appears to be nonsense, returning empty response")
            synapse.veridex_response = []
            return synapse
        
        # Get evidence using multiple parallel search methods
        evidence_list = []
        
        # Create event loop and run async searches
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            evidence_list = loop.run_until_complete(
                self.gather_evidence_async(statement, request_id, synapse.sources)
            )
        except Exception as e:
            bt.logging.error(f"{request_id} | Error gathering evidence: {e}")
            traceback.print_exc()
        
        # Filter and sort the evidence
        final_evidence = self.filter_and_prepare_evidence(evidence_list, statement)
        
        synapse.veridex_response = final_evidence
        elapsed = time.time() - start_time
        bt.logging.info(
            f"{request_id} | Returning {len(final_evidence)} evidence items in {elapsed:.2f}s for: '{statement}'"
        )
        
        return synapse
    
    def is_likely_nonsense(self, statement: str) -> bool:
        """Check if a statement appears to be nonsense"""
        statement_lower = statement.lower()
        
        # Check for nonsense keywords
        for keyword in NONSENSE_KEYWORDS:
            if keyword.lower() in statement_lower:
                return True
                
        # Check for implausible claims (statements with extreme or impossible quantities)
        implausible_patterns = [
            r'\b(all|every|no)\s+human',
            r'\bnever\b.*\bhistory\b',
            r'\balways\b.*\bhistory\b',
            r'\b100%\s+(of|certain|sure)',
        ]
        
        for pattern in implausible_patterns:
            if re.search(pattern, statement_lower):
                return True
        
        return False
        
    async def gather_evidence_async(self, statement: str, request_id: str, preferred_sources: List[str]) -> List[Dict]:
        """Coordinate multiple async search methods to gather evidence"""
        session = await self.create_aiohttp_session()
        
        search_tasks = []
        
        # Add search method tasks based on available APIs
        if self.perplexity_client:
            search_tasks.append(self.call_perplexity_ai_async(statement))
        
        if self.google_api_key and self.google_cse_id:
            search_tasks.append(self.google_search_async(statement, session))
        
        if self.bing_api_key:
            search_tasks.append(self.bing_search_async(statement, session))
        
        # Always include web search as fallback
        search_tasks.append(self.web_search_async(statement, session))
        
        # If preferred sources are provided, add them as additional tasks
        if preferred_sources:
            for source in preferred_sources:
                if self.is_valid_url(source):
                    search_tasks.append(self.extract_from_specific_url_async(statement, source, session))
        
        # Add trusted domain searches for diversity
        random_trusted_domains = random.sample(TRUSTED_DOMAINS, min(5, len(TRUSTED_DOMAINS)))
        for domain in random_trusted_domains:
            search_query = f"site:{domain} {statement}"
            search_tasks.append(self.web_search_async(search_query, session))
        
        # Run all search tasks concurrently with timeout
        try:
            timeout = self.config.async_timeout
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception as e:
            bt.logging.error(f"{request_id} | Error in gather_evidence_async: {e}")
            results = []
        
        # Flatten and filter results
        all_evidence = []
        for result in results:
            if isinstance(result, list):
                all_evidence.extend(result)
            elif isinstance(result, Exception):
                bt.logging.warning(f"Search error: {result}")
        
        return all_evidence
    
    def filter_and_prepare_evidence(self, evidence_list: List[Dict], statement: str) -> List[SourceEvidence]:
        """Process all evidence to create the final response"""
        # Remove duplicates and invalid entries
        filtered_evidence = []
        seen_urls = set()
        seen_excerpts = set()
        
        # First sort by domain diversity to prioritize diverse sources
        random.shuffle(evidence_list)  # Introduce randomness
        
        # Process evidence items
        for item in evidence_list:
            try:
                url = item.get("url", "").strip()
                snippet = item.get("snippet", "").strip()
                
                if not url or not snippet or len(snippet) < 20:
                    continue
                
                # Skip if URL or snippet already seen (deduplication)
                if url in seen_urls or snippet in seen_excerpts:
                    continue
                
                # Parse domain and check if already used too many times
                domain = self.extract_domain(url)
                if not domain:
                    continue
                
                # Add to tracking sets
                seen_urls.add(url)
                seen_excerpts.add(snippet)
                self.current_domains.add(domain)
                
                # Create evidence object
                evidence = {
                    "url": url,
                    "snippet": snippet,
                    "domain": domain
                }
                
                filtered_evidence.append(evidence)
                
                # Limit number of evidence items to prevent overloading
                if len(filtered_evidence) >= self.config.max_snippets:
                    break
                    
            except Exception as e:
                bt.logging.error(f"Error processing evidence item: {e}")
        
        # Convert to SourceEvidence objects
        result = []
        for evidence in filtered_evidence:
            source_evidence = SourceEvidence(
                url=evidence["url"],
                excerpt=evidence["snippet"]
            )
            result.append(source_evidence)
            
        return result
            
    def extract_domain(self, url: str) -> str:
        """Extract the base domain from a URL"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except:
            return ""
    
    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def call_perplexity_ai_async(self, statement: str) -> List[dict]:
        """
        Use Perplexity AI to find relevant evidence.
        """
        if not self.perplexity_client:
            return []
            
        system_content = """
You are an API that fact checks statements.

Rules:
1. Return the response **only as a JSON array**.
2. The response **must be a valid JSON array**, formatted as:
   ```json
   [{"url": "<source url>", "snippet": "<snippet that directly agrees with or contradicts statement>"}]
3. Do not include any introductory text, explanations, or additional commentary.
4. Do not add any labels, headers, or markdown formatting—only return the JSON array.

Steps:
1. Find sources / text segments that either contradict or agree with the user provided statement.
2. Pick and extract the segments that most strongly agree or contradict the statement.
3. Do not return urls or segments that do not directly support or disagree with the statement.
4. Do not change any text in the segments (must return an exact html text match), but do shorten the segment to get only the part that directly agrees or disagrees with the statement.
5. Use diverse and credible sources - prioritize different domains.
6. For each search result, use only one snippet per domain.
7. Create the json object for each source and statement and add them only INTO ONE array

Response MUST returned as a json array. If it isn't returned as json object the response MUST BE EMPTY.
"""
        user_content = f"Return snippets that strongly agree with or reject the following statement:\n{statement}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw_text = None
        try:
            response = self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=False
            )
            if not hasattr(response, "choices") or len(response.choices) == 0:
                bt.logging.warn(f"Perplexity returned no choices")
                return []
                
            raw_text = response.choices[0].message.content.strip()
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()

            data = json.loads(raw_text)
            if not isinstance(data, list):
                bt.logging.warn(f"Perplexity response is not a list")
                return []
                
            return data
        except Exception as e:
            if raw_text is not None:
                bt.logging.error(f"Raw Text of AI Response: {raw_text}")

            bt.logging.error(f"Error calling Perplexity AI: {e}")
            return []
    
    async def google_search_async(self, query: str, session) -> List[dict]:
        """Search using Google Custom Search API"""
        if not self.google_api_key or not self.google_cse_id:
            return []
            
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": 10
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                items = data.get("items", [])
                
                results = []
                for item in items:
                    search_url = item.get("link")
                    snippet = item.get("snippet", "")
                    
                    if search_url and snippet:
                        # Check if domain already used
                        domain = self.extract_domain(search_url)
                        if domain in self.current_domains:
                            continue
                            
                        # Get full text from the URL
                        full_text = await self.extract_text_from_url_async(search_url, session)
                        if full_text:
                            # Find the best match in the full text
                            better_snippet = self.find_relevant_snippet(full_text, query, snippet)
                            if better_snippet:
                                results.append({
                                    "url": search_url,
                                    "snippet": better_snippet
                                })
                
                return results
                
        except Exception as e:
            bt.logging.error(f"Google search error: {e}")
            return []
    
    async def bing_search_async(self, query: str, session) -> List[dict]:
        """Search using Bing Search API"""
        if not self.bing_api_key:
            return []
            
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        params = {"q": query, "count": 10, "responseFilter": "Webpages"}
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                webpages = data.get("webPages", {}).get("value", [])
                
                results = []
                for page in webpages:
                    search_url = page.get("url")
                    snippet = page.get("snippet", "")
                    
                    if search_url and snippet:
                        # Check if domain already used
                        domain = self.extract_domain(search_url)
                        if domain in self.current_domains:
                            continue
                            
                        # Get full text from the URL
                        full_text = await self.extract_text_from_url_async(search_url, session)
                        if full_text:
                            # Find the best match in the full text
                            better_snippet = self.find_relevant_snippet(full_text, query, snippet)
                            if better_snippet:
                                results.append({
                                    "url": search_url,
                                    "snippet": better_snippet
                                })
                
                return results
                
        except Exception as e:
            bt.logging.error(f"Bing search error: {e}")
            return []
    
    async def web_search_async(self, query: str, session) -> List[dict]:
        """Fallback search method using a web search engine"""
        # Use DuckDuckGo API-less search as fallback
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        try:
            async with session.get(search_url) as response:
                if response.status != 200:
                    return []
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                for result in soup.select('.result'):
                    title_elem = result.select_one('.result__a')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if not title_elem or not snippet_elem:
                        continue
                        
                    # Extract URL from href attribute
                    url_elem = title_elem.get('href', '')
                    if not url_elem:
                        continue
                        
                    # Parse the URL parameter
                    try:
                        parsed_url = urllib.parse.parse_qs(url_elem.split('?')[1])
                        result_url = parsed_url.get('uddg', [''])[0]
                    except:
                        continue
                        
                    if not result_url:
                        continue
                        
                    snippet = snippet_elem.get_text().strip()
                    
                    # Check if domain already used
                    domain = self.extract_domain(result_url)
                    if domain in self.current_domains:
                        continue
                        
                    # Get full text from the URL
                    full_text = await self.extract_text_from_url_async(result_url, session)
                    if full_text:
                        # Find the best match in the full text
                        better_snippet = self.find_relevant_snippet(full_text, query, snippet)
                        if better_snippet:
                            results.append({
                                "url": result_url,
                                "snippet": better_snippet
                            })
                    
                    if len(results) >= 10:
                        break
                        
                return results
                
        except Exception as e:
            bt.logging.error(f"Web search error: {e}")
            return []
    
    async def extract_from_specific_url_async(self, statement: str, url: str, session) -> List[dict]:
        """Extract relevant snippets from a specific URL"""
        try:
            full_text = await self.extract_text_from_url_async(url, session)
            if not full_text:
                return []
                
            # Find the best match in the full text
            snippet = self.find_relevant_snippet(full_text, statement)
            if not snippet:
                return []
                
            return [{
                "url": url,
                "snippet": snippet
            }]
            
        except Exception as e:
            bt.logging.error(f"Error extracting from URL {url}: {e}")
            return []
    
    async def extract_text_from_url_async(self, url: str, session) -> str:
        """Extract text content from a URL"""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return ""
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text = soup.get_text()
                
                # Break into lines and remove leading and trailing space
                lines = (line.strip() for line in text.splitlines())
                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # Remove blank lines
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text
        except Exception as e:
            bt.logging.error(f"Error extracting text from {url}: {e}")
            return ""
    
    def find_relevant_snippet(self, text: str, query: str, fallback_snippet: str = "") -> str:
        """Find the most relevant snippet in a text for a query"""
        if not text:
            return fallback_snippet
            
        # Clean query for better matching
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        words = clean_query.split()
        
        # Split text into sentences or paragraphs
        paragraphs = re.split(r'(?<=[.!?])\s+', text)
        
        best_score = 0
        best_snippet = fallback_snippet
        
        for paragraph in paragraphs:
            if len(paragraph) < 20 or len(paragraph) > 1000:
                continue
                
            # Calculate relevance score
            score = 0
            for word in words:
                if word in paragraph.lower():
                    score += 1
            
            # Normalize by paragraph length
            score = score / (len(paragraph.split()) ** 0.5)
            
            if score > best_score:
                best_score = score
                best_snippet = paragraph
        
        # If no good match found, return fallback
        if best_score == 0:
            return fallback_snippet
            
        return best_snippet.strip()
    
    def setup_axon(self):
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(f"Attaching forward function to axon")
        self.axon.attach(
            forward_fn=self.veridex_forward,
            blacklist_fn=self.blacklist_fn,
        )
        bt.logging.info(f"Serving axon on network: {self.config.subtensor.network} netuid: {self.config.netuid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Axon: {self.axon}")

        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

    async def close_session(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def run(self):
        bt.logging.info("Setting up axon")
        self.setup_axon()

        bt.logging.info("Setting up proxy logger")
        self.setup_proxy_logger()

        bt.logging.info("Starting main loop")
        step = 0
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        while True:
            try:
                if step % 60 == 0:
                    self.metagraph.sync()
                    log = (f"Block: {self.metagraph.block.item()} | "
                           f"Incentive: {self.metagraph.I[self.my_subnet_uid]} | ")
                    bt.logging.info(log)
                step += 1
                time.sleep(1)
            except KeyboardInterrupt:
                self.axon.stop()
                loop.run_until_complete(self.close_session())
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    miner = CustomMiner()
    miner.run()