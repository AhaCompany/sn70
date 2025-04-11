import os
import time
import argparse
import traceback
import bittensor as bt
import json
import random
import re
from typing import Tuple, List, Dict
import logging
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from shared.log_data import LoggerType
from shared.proxy_log_handler import register_proxy_log_handler
from shared.veridex_protocol import VericoreSynapse, SourceEvidence

# debug
bt.logging.set_trace()

load_dotenv()

# Trusted domains for high-quality sources
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

class OpenAIMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_bittensor_objects()
        self.setup_logging()
        
        # Initialize OpenAI client
        self.setup_openai()

    def setup_openai(self):
        """Set up OpenAI API client"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not self.openai_api_key:
            bt.logging.error("No OPENAI_API_KEY found in environment. Please set it to use this miner.")
            exit(1)
            
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            bt.logging.info("OpenAI client initialized successfully")
        except Exception as e:
            bt.logging.error(f"Failed to initialize OpenAI client: {e}")
            exit(1)

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID.")
        parser.add_argument("--max_snippets", type=int, default=5, help="Maximum snippets to return per request")
        parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
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
                "openai_miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running OpenAI miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
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
        
        statement = synapse.statement
        
        # Check if statement is likely nonsense
        if self.is_likely_nonsense(statement):
            bt.logging.info(f"{request_id} | Statement appears to be nonsense, returning empty response")
            synapse.veridex_response = []
            return synapse
        
        # Get evidence using OpenAI
        evidence_list = self.call_openai(statement, synapse.sources)
        
        # Filter and prepare the final evidence list
        final_evidence = []
        seen_domains = set()
        
        for item in evidence_list:
            url = item.get("url", "").strip()
            snippet = item.get("snippet", "").strip()
            
            if not url or not snippet or len(snippet) < 20:
                continue
                
            # Check domain diversity
            domain = self.extract_domain(url)
            if domain in seen_domains:
                continue
                
            seen_domains.add(domain)
            
            # Add to final evidence
            ev = SourceEvidence(url=url, excerpt=snippet)
            final_evidence.append(ev)
            
            # Limit to max snippets
            if len(final_evidence) >= self.config.max_snippets:
                break
        
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
                
        # Check for implausible claims
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
    
    def call_openai(self, statement: str, preferred_sources: List[str]) -> List[dict]:
        """
        Use OpenAI to find relevant evidence for the statement.
        """
        # Build a strong system prompt
        system_prompt = f"""You are a fact-checking expert specialized in finding quality evidence from diverse, reputable sources.

YOUR TASK:
Find evidence that either STRONGLY SUPPORTS or STRONGLY CONTRADICTS this statement: "{statement}"

REQUIREMENTS:
1. Respond ONLY with a JSON array in this exact format: 
   [{{
     "url": "full_url_to_source",
     "snippet": "exact_text_from_source_that_supports_or_contradicts"
   }}]

2. Each snippet MUST:
   - Be an EXACT QUOTE from the source (not paraphrased)
   - Strongly confirm OR strongly contradict the statement
   - Be 1-3 sentences focused on the key evidence
   - Come from a legitimate, verifiable source

3. For maximum score:
   - Use DIVERSE domains - each evidence should come from a different domain
   - Prioritize highly credible sources (academic, government, established news)
   - Include 3-5 pieces of strong evidence
   - Avoid neutral or tangentially related content
   - Return BOTH supporting AND contradicting evidence if available

4. NEVER fabricate evidence or URLs

5. If the statement appears nonsensical or impossible to verify with facts, return an empty array: []

SUGGESTED DOMAINS:
{", ".join(random.sample(TRUSTED_DOMAINS, min(10, len(TRUSTED_DOMAINS))))}

Respond ONLY with the JSON array, no other text.
"""

        user_prompt = f"Find strong evidence for or against this statement. If preferred sources are provided, use them: {', '.join(preferred_sources) if preferred_sources else 'No specific sources provided'}"
        
        try:
            # Send request to OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more factual responses
                response_format={"type": "json_object"},
                timeout=15  # 15 second timeout for faster responses
            )
            
            # Extract and parse response
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                # Check if the data is the expected format (list of objects with url and snippet)
                if isinstance(data, dict) and "results" in data:
                    return data["results"]
                elif isinstance(data, list):
                    return data
                else:
                    bt.logging.warning(f"Unexpected response format from OpenAI: {data}")
                    return []
            except json.JSONDecodeError:
                bt.logging.error(f"Failed to parse JSON from OpenAI response: {content}")
                return []
                
        except Exception as e:
            bt.logging.error(f"Error calling OpenAI: {e}")
            return []
    
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

    def run(self):
        bt.logging.info("Setting up axon")
        self.setup_axon()

        bt.logging.info("Setting up proxy logger")
        self.setup_proxy_logger()

        bt.logging.info("Starting main loop")
        step = 0
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
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    miner = OpenAIMiner()
    miner.run()