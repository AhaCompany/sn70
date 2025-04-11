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
        bt.logging.info(f"{request_id} | === NEW REQUEST === Statement: '{synapse.statement}'")
        bt.logging.info(f"{request_id} | Preferred sources: {synapse.sources}")
        
        statement = synapse.statement
        
        # Check if statement is likely nonsense
        if self.is_likely_nonsense(statement):
            bt.logging.info(f"{request_id} | Statement appears to be nonsense, returning empty response")
            synapse.veridex_response = []
            return synapse
        
        # Get evidence using OpenAI
        bt.logging.info(f"{request_id} | Calling OpenAI to fetch evidence")
        evidence_list = self.call_openai(statement, synapse.sources)
        bt.logging.info(f"{request_id} | Raw evidence count: {len(evidence_list)}")
        
        if evidence_list:
            bt.logging.info(f"{request_id} | Sample evidence: {evidence_list[0]}")
        else:
            bt.logging.warning(f"{request_id} | No evidence returned from OpenAI")
        
        # Filter and prepare the final evidence list
        final_evidence = []
        seen_domains = set()
        
        bt.logging.info(f"{request_id} | Filtering and preparing evidence")
        
        for idx, item in enumerate(evidence_list):
            url = item.get("url", "").strip()
            snippet = item.get("snippet", "").strip()
            
            if not url or not snippet:
                bt.logging.warning(f"{request_id} | Item {idx} missing URL or snippet")
                continue
                
            if len(snippet) < 20:
                bt.logging.warning(f"{request_id} | Item {idx} snippet too short: '{snippet}'")
                continue
                
            # Check domain diversity
            domain = self.extract_domain(url)
            if domain in seen_domains:
                bt.logging.info(f"{request_id} | Skipping duplicate domain: {domain}")
                continue
                
            seen_domains.add(domain)
            bt.logging.info(f"{request_id} | Using evidence from domain: {domain}")
            
            # Add to final evidence
            ev = SourceEvidence(url=url, excerpt=snippet)
            final_evidence.append(ev)
            bt.logging.info(f"{request_id} | Added evidence {len(final_evidence)}: URL={url}, Excerpt={snippet[:50]}...")
            
            # Limit to max snippets
            if len(final_evidence) >= self.config.max_snippets:
                bt.logging.info(f"{request_id} | Reached max snippets ({self.config.max_snippets}), stopping")
                break
        
        synapse.veridex_response = final_evidence
        elapsed = time.time() - start_time
        bt.logging.info(
            f"{request_id} | === RESPONSE COMPLETE === Returning {len(final_evidence)} evidence items in {elapsed:.2f}s"
        )
        
        # Log all evidence for debugging
        for i, ev in enumerate(final_evidence):
            bt.logging.info(f"{request_id} | Evidence {i+1}: {ev.url} - {ev.excerpt[:50]}...")
        
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
        request_id = str(int(time.time()))
        bt.logging.info(f"{request_id} | [OpenAI] Starting evidence search for: '{statement[:50]}...'")
        
        # Build a strong system prompt
        system_prompt = f"""You are a fact-checking expert specialized in finding quality evidence from diverse, reputable sources.

YOUR TASK:
Find evidence that either STRONGLY SUPPORTS or STRONGLY CONTRADICTS this statement: "{statement}"

REQUIREMENTS:
1. You MUST respond with ONLY a JSON array in this exact format: 
   [
     {{
       "url": "full_url_to_source",
       "snippet": "exact_text_from_source_that_supports_or_contradicts"
     }},
     {{
       "url": "another_url",
       "snippet": "another_snippet"
     }}
   ]

2. The response MUST be a valid JSON array containing objects, not a single object.

3. Each snippet MUST:
   - Be an EXACT QUOTE from the source (not paraphrased)
   - Strongly confirm OR strongly contradict the statement
   - Be 1-3 sentences focused on the key evidence
   - Come from a legitimate, verifiable source

4. For maximum score:
   - Use DIVERSE domains - each evidence should come from a different domain
   - Prioritize highly credible sources (academic, government, established news)
   - Include 3-5 pieces of strong evidence
   - Avoid neutral or tangentially related content
   - Return BOTH supporting AND contradicting evidence if available

5. NEVER fabricate evidence or URLs

6. If the statement appears nonsensical or impossible to verify with facts, return an empty array: []

SUGGESTED DOMAINS:
{", ".join(random.sample(TRUSTED_DOMAINS, min(10, len(TRUSTED_DOMAINS))))}

Respond ONLY with the JSON array, no other text.
"""

        user_prompt = f"Find strong evidence for or against this statement. If preferred sources are provided, use them: {', '.join(preferred_sources) if preferred_sources else 'No specific sources provided'}"
        
        bt.logging.info(f"{request_id} | [OpenAI] Using model: {self.config.model}")
        
        try:
            # Log configuration before calling OpenAI
            response_format_type = "json_array" if self.config.model in ["gpt-4o", "gpt-4-turbo", "gpt-4-0125"] else "json_object"
            bt.logging.info(f"{request_id} | [OpenAI] Using response_format: {response_format_type}")
            
            # Send request to OpenAI
            request_start_time = time.time()
            bt.logging.info(f"{request_id} | [OpenAI] Sending request to API")
            
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more factual responses
                # Use json_array instead of json_object for format
                response_format={"type": response_format_type},
                timeout=15  # 15 second timeout for faster responses
            )
            
            request_duration = time.time() - request_start_time
            bt.logging.info(f"{request_id} | [OpenAI] Received response in {request_duration:.2f}s")
            
            # Extract and parse response
            content = response.choices[0].message.content
            bt.logging.info(f"{request_id} | [OpenAI] Raw response: {content[:500]}...")
            
            try:
                data = json.loads(content)
                bt.logging.info(f"{request_id} | [OpenAI] Successfully parsed JSON response")
                
                # Handle different response formats
                if isinstance(data, dict) and "results" in data:
                    # Format: {"results": [{...}, {...}]}
                    results = data["results"]
                    bt.logging.info(f"{request_id} | [OpenAI] Found results key in response dictionary")
                    
                    if isinstance(results, list):
                        bt.logging.info(f"{request_id} | [OpenAI] Results is a list with {len(results)} items")
                        return results
                    elif isinstance(results, dict):
                        # Single result in a dict
                        bt.logging.info(f"{request_id} | [OpenAI] Results is a single dict, converting to list")
                        return [results]
                    else:
                        bt.logging.warning(f"{request_id} | [OpenAI] Unexpected results format: {type(results)}")
                        return []
                
                elif isinstance(data, dict) and "url" in data and "snippet" in data:
                    # Format: {"url": "...", "snippet": "..."}
                    # Single object instead of array - convert to array
                    bt.logging.info(f"{request_id} | [OpenAI] Response is a single evidence object, converting to array")
                    return [data]
                
                elif isinstance(data, list):
                    # Format: [{...}, {...}]
                    # Already an array
                    bt.logging.info(f"{request_id} | [OpenAI] Response is already an array with {len(data)} items")
                    
                    # Validate array contents
                    valid_items = []
                    for i, item in enumerate(data):
                        if isinstance(item, dict) and "url" in item and "snippet" in item:
                            valid_items.append(item)
                        else:
                            bt.logging.warning(f"{request_id} | [OpenAI] Item {i} in array is invalid: {item}")
                    
                    bt.logging.info(f"{request_id} | [OpenAI] Found {len(valid_items)} valid items in array")
                    return valid_items
                
                else:
                    # Unexpected format
                    bt.logging.warning(f"{request_id} | [OpenAI] Unexpected response format: {type(data)}")
                    
                    if isinstance(data, dict):
                        bt.logging.warning(f"{request_id} | [OpenAI] Dictionary keys: {list(data.keys())}")
                        
                        if not data:  # Empty dict
                            bt.logging.warning(f"{request_id} | [OpenAI] Empty dictionary returned")
                            return []
                        
                        # Try to find any url/snippet pairs
                        result = []
                        for key, value in data.items():
                            if isinstance(value, dict) and "url" in value and "snippet" in value:
                                bt.logging.info(f"{request_id} | [OpenAI] Found nested evidence in key: {key}")
                                result.append(value)
                        
                        if result:
                            bt.logging.info(f"{request_id} | [OpenAI] Extracted {len(result)} evidence items from dictionary")
                        
                        return result
                    
                    return []
                
            except json.JSONDecodeError as decode_error:
                bt.logging.error(f"{request_id} | [OpenAI] Failed to parse JSON: {decode_error}")
                bt.logging.error(f"{request_id} | [OpenAI] Raw content: {content}")
                
                # Try to fix and extract JSON if possible
                try:
                    # Look for JSON array pattern
                    json_match = re.search(r'\[\s*{.+}\s*\]', content, re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(0)
                        bt.logging.info(f"{request_id} | [OpenAI] Attempting to parse extracted JSON: {potential_json[:100]}...")
                        extracted_data = json.loads(potential_json)
                        if isinstance(extracted_data, list):
                            bt.logging.info(f"{request_id} | [OpenAI] Successfully extracted JSON array with {len(extracted_data)} items")
                            return extracted_data
                except Exception as extract_error:
                    bt.logging.error(f"{request_id} | [OpenAI] Failed to extract JSON: {extract_error}")
                
                return []
                
        except Exception as e:
            bt.logging.error(f"{request_id} | [OpenAI] Error calling OpenAI: {e}")
            bt.logging.error(f"{request_id} | [OpenAI] Error details: {traceback.format_exc()}")
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