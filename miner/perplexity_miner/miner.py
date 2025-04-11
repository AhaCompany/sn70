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

class PerplexityMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_bittensor_objects()
        self.setup_logging()
        
        # Initialize Perplexity client
        self.setup_perplexity()

    def setup_perplexity(self):
        """Set up Perplexity API client"""
        self.perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # Try to match original miner using sonar-pro
        self.perplexity_models = [
            "sonar-pro",            # Main model from original miner (first try this)
            "sonar-medium-chat",    # Alternative model 
            "sonar-small-chat",     # Alternative model
            "pplx-7b-online",       # New model with online search
            "pplx-70b-online",      # New model with online search
            "mistral-7b-instruct",  # Mistral model
            "claude-instant",       # Claude model on Perplexity
        ]
        self.default_model = "sonar-pro"  # Try to match original miner
        
        # First try Perplexity
        if self.perplexity_api_key:
            try:
                self.perplexity_client = OpenAI(
                    api_key=self.perplexity_api_key,
                    base_url="https://api.perplexity.ai"
                )
                bt.logging.info("Perplexity client initialized successfully")
                
                # Try each model until we find one that works
                for model in self.perplexity_models:
                    try:
                        bt.logging.info(f"Testing Perplexity with model: {model}")
                        test_response = self.perplexity_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "hello"}],
                            max_tokens=5
                        )
                        # If we get here, the model worked
                        self.default_model = model
                        bt.logging.info(f"Perplexity connection test successful with model: {model}")
                        break
                    except Exception as model_e:
                        bt.logging.warning(f"Model {model} not available: {model_e}")
                        continue
                
                # Check if we found a working model
                if not hasattr(self, 'default_model') or not self.default_model:
                    bt.logging.error("No working Perplexity models found")
                    self.perplexity_client = None
                    
            except Exception as e:
                bt.logging.error(f"Perplexity API initialization failed: {e}")
                self.perplexity_client = None
        else:
            bt.logging.warning("No PERPLEXITY_API_KEY found")
            self.perplexity_client = None
            
        # Fall back to OpenAI if Perplexity fails
        if not self.perplexity_client and self.openai_api_key:
            try:
                bt.logging.info("Falling back to OpenAI client")
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                bt.logging.info("OpenAI client initialized successfully")
            except Exception as e:
                bt.logging.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            
        # Exit if we have no working API clients
        if not self.perplexity_client and not self.openai_client:
            bt.logging.error("No working API clients. Please check your API keys.")
            exit(1)

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID.")
        parser.add_argument("--max_snippets", type=int, default=5, help="Maximum snippets to return per request")
        parser.add_argument("--model", type=str, default="sonar-pro", help="Perplexity model to use")
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
                "perplexity_miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running Perplexity miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
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
        
        # Get evidence using Perplexity with fallback to OpenAI
        bt.logging.info(f"{request_id} | Calling API services to fetch evidence")
        evidence_list = self.call_perplexity_ai(statement, synapse.sources)
        bt.logging.info(f"{request_id} | Raw evidence count: {len(evidence_list)}")
        
        if evidence_list:
            bt.logging.info(f"{request_id} | Sample evidence: {evidence_list[0]}")
        else:
            bt.logging.warning(f"{request_id} | No evidence returned from APIs")
        
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
    
    def call_perplexity_ai(self, statement: str, preferred_sources: List[str]) -> List[dict]:
        """
        Use Perplexity AI or fallback to OpenAI to find relevant evidence for the statement.
        """
        results = []
        
        # Try Perplexity first if available
        if self.perplexity_client:
            results = self._call_perplexity(statement, preferred_sources)
            bt.logging.info(f"Perplexity returned {len(results)} results")
        
        # Fall back to OpenAI if Perplexity failed
        if not results and self.openai_client:
            bt.logging.info("Falling back to OpenAI")
            results = self._call_openai(statement, preferred_sources)
            bt.logging.info(f"OpenAI returned {len(results)} results")
            
        return results
    
    def _call_perplexity(self, statement: str, preferred_sources: List[str]) -> List[dict]:
        """
        Use Perplexity AI to find evidence
        """
        request_id = str(int(time.time()))
        bt.logging.info(f"{request_id} | [Perplexity] Starting evidence search for: '{statement[:50]}...'")
        
        # Build a strong system prompt
        system_content = """
You are an expert fact-checking system that finds high-quality evidence from diverse sources.

YOUR MISSION:
Find evidence that either STRONGLY SUPPORTS or STRONGLY CONTRADICTS the user's statement.

RESPONSE FORMAT:
Return ONLY a JSON array with this exact structure:
[{
  "url": "full_url_to_source",
  "snippet": "exact_text_from_source_that_supports_or_contradicts"
}]

REQUIREMENTS:
1. Each snippet MUST:
   - Be an EXACT QUOTE from the source (can be verified by the validator)
   - Either strongly support or strongly contradict the statement
   - Be concise (1-3 sentences) and directly relevant
   - Come from a legitimate, verifiable source

2. For maximum score:
   - Use DIVERSE domains - never use the same domain twice
   - Prioritize highly credible sources
   - Include both supporting AND contradicting evidence if available
   - Return 3-5 high-quality pieces of evidence
   - Avoid neutral or tangentially related content

3. NEVER fabricate quotes or URLs

4. If the statement is nonsensical or impossible to verify with factual sources, return an empty array: []

DO NOT include any text outside the JSON array - only return the JSON array itself.
"""

        # Add preferred sources if provided
        source_prompt = ""
        if preferred_sources:
            source_prompt = f"\n\nPreferred sources to check first: {', '.join(preferred_sources)}"
        
        user_content = f"Return snippets that strongly agree with or reject the following statement:\n{statement}{source_prompt}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw_text = None
        try:
            # Use sonar-pro just like the original miner
            try_model = "sonar-pro"
            bt.logging.info(f"{request_id} | [Perplexity] Using model: {try_model}")
            
            # Send request to Perplexity
            request_start_time = time.time()
            bt.logging.info(f"{request_id} | [Perplexity] Sending request to API")
            
            # Match the original miner's configuration exactly
            try:
                response = self.perplexity_client.chat.completions.create(
                    model=try_model,
                    messages=messages,
                    stream=False
                )
            except Exception as model_error:
                # If sonar-pro fails, try the next best model
                bt.logging.warning(f"{request_id} | [Perplexity] Model sonar-pro failed: {model_error}")
                # Try sonar-small-chat as fallback
                bt.logging.info(f"{request_id} | [Perplexity] Trying fallback model: sonar-small-chat")
                response = self.perplexity_client.chat.completions.create(
                    model="sonar-small-chat",
                    messages=messages,
                    stream=False
                )
            
            request_duration = time.time() - request_start_time
            bt.logging.info(f"{request_id} | [Perplexity] Received response in {request_duration:.2f}s")
            
            if not hasattr(response, "choices") or len(response.choices) == 0:
                bt.logging.warn(f"{request_id} | [Perplexity] API returned no choices")
                return []
                
            raw_text = response.choices[0].message.content.strip()
            bt.logging.info(f"{request_id} | [Perplexity] Raw response (first 200 chars): {raw_text[:200]}...")
            
            # Remove markdown code blocks if present
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()
            bt.logging.info(f"{request_id} | [Perplexity] After removing markdown (first 100 chars): {raw_text[:100]}...")

            # Try to parse the JSON, possibly fixing minor issues
            try:
                data = json.loads(raw_text)
                bt.logging.info(f"{request_id} | [Perplexity] Successfully parsed JSON response")
                
            except json.JSONDecodeError as json_err:
                # Log the error and try to fix common issues
                bt.logging.warning(f"{request_id} | [Perplexity] JSON parse error: {json_err}, attempting to fix")
                
                # Sometimes content has extra backticks or is wrapped in code blocks
                if "```" in raw_text:
                    # Extract just the JSON part
                    bt.logging.info(f"{request_id} | [Perplexity] Detected code blocks in response, extracting JSON")
                    try:
                        json_part = re.search(r'```(?:json)?(.*?)```', raw_text, re.DOTALL)
                        if json_part:
                            fixed_text = json_part.group(1).strip()
                            bt.logging.info(f"{request_id} | [Perplexity] Extracted JSON part: {fixed_text[:100]}...")
                            data = json.loads(fixed_text)
                            bt.logging.info(f"{request_id} | [Perplexity] Successfully parsed extracted JSON")
                        else:
                            bt.logging.error(f"{request_id} | [Perplexity] Failed to extract JSON from code blocks")
                            raise json.JSONDecodeError("Failed to extract JSON", raw_text, 0)
                    except Exception as e:
                        bt.logging.error(f"{request_id} | [Perplexity] Failed to fix JSON: {e}")
                        return []
                else:
                    # Try to extract any JSON array from the response
                    bt.logging.info(f"{request_id} | [Perplexity] Trying to extract JSON array from response")
                    try:
                        # Look for JSON array pattern
                        json_match = re.search(r'\[\s*{.+}\s*\]', raw_text, re.DOTALL)
                        if json_match:
                            potential_json = json_match.group(0)
                            bt.logging.info(f"{request_id} | [Perplexity] Found potential JSON array: {potential_json[:100]}...")
                            data = json.loads(potential_json)
                            bt.logging.info(f"{request_id} | [Perplexity] Successfully extracted and parsed JSON array")
                        else:
                            bt.logging.error(f"{request_id} | [Perplexity] Could not find JSON array pattern in response")
                            return []
                    except Exception as e:
                        bt.logging.error(f"{request_id} | [Perplexity] Could not fix JSON format: {e}")
                        bt.logging.error(f"{request_id} | [Perplexity] Raw text: {raw_text}")
                        return []
            
            # Process the data
            bt.logging.info(f"{request_id} | [Perplexity] Processing parsed data type: {type(data)}")
            
            if isinstance(data, dict) and "results" in data:
                # If data is in format {"results": [...]}
                results = data["results"]
                bt.logging.info(f"{request_id} | [Perplexity] Found results key in response dictionary")
                
                if isinstance(results, list):
                    bt.logging.info(f"{request_id} | [Perplexity] Results is a list with {len(results)} items")
                    return results
                elif isinstance(results, dict) and "url" in results and "snippet" in results:
                    # Single result in a dict
                    bt.logging.info(f"{request_id} | [Perplexity] Results is a single dict, converting to list")
                    return [results]
                else:
                    bt.logging.warning(f"{request_id} | [Perplexity] Unexpected results format: {type(results)}")
                    return []
                    
            elif isinstance(data, dict) and "url" in data and "snippet" in data:
                # If data is a single object instead of an array
                bt.logging.info(f"{request_id} | [Perplexity] Response is a single evidence object, converting to array")
                return [data]
                
            elif isinstance(data, list):
                # Already an array
                bt.logging.info(f"{request_id} | [Perplexity] Response is already an array with {len(data)} items")
                
                # Validate array contents
                valid_items = []
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "url" in item and "snippet" in item:
                        valid_items.append(item)
                    else:
                        bt.logging.warning(f"{request_id} | [Perplexity] Item {i} in array is invalid: {item}")
                
                bt.logging.info(f"{request_id} | [Perplexity] Found {len(valid_items)} valid items in array")
                return valid_items
                
            else:
                bt.logging.warning(f"{request_id} | [Perplexity] Response is not in expected format: {type(data)}")
                return []
                
        except Exception as e:
            if 'raw_text' in locals() and raw_text is not None:
                bt.logging.error(f"{request_id} | [Perplexity] Raw Text of API Response: {raw_text}")

            bt.logging.error(f"{request_id} | [Perplexity] Error calling Perplexity AI: {e}")
            bt.logging.error(f"{request_id} | [Perplexity] Error details: {traceback.format_exc()}")
            return []
            
    def _call_openai(self, statement: str, preferred_sources: List[str]) -> List[dict]:
        """
        Use OpenAI to find evidence when Perplexity fails
        """
        if not self.openai_client:
            return []
            
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

Respond ONLY with the JSON array, no other text.
"""

        # Add preferred sources if provided
        sources_text = ""
        if preferred_sources:
            sources_text = f" If preferred sources are provided, use them: {', '.join(preferred_sources)}"
            
        user_prompt = f"Find strong evidence for or against this statement.{sources_text}"
        
        try:
            # Use gpt-3.5-turbo which is faster and cheaper than gpt-4
            bt.logging.info("Using OpenAI model: gpt-3.5-turbo")
            
            # Send request to OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",  # Use a consistent, stable model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more factual responses
                response_format={"type": "json_array"},
                timeout=15  # 15 second timeout for faster responses
            )
            
            # Extract and parse response
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                
                # Handle different response formats
                if isinstance(data, dict) and "results" in data:
                    # Format: {"results": [{...}, {...}]}
                    results = data["results"]
                    if isinstance(results, list):
                        return results
                    elif isinstance(results, dict):
                        # Single result in a dict
                        return [results]
                    else:
                        bt.logging.warning(f"Unexpected results format: {results}")
                        return []
                
                elif isinstance(data, dict) and "url" in data and "snippet" in data:
                    # Format: {"url": "...", "snippet": "..."}
                    # Single object instead of array - convert to array
                    bt.logging.info(f"Converting single object to array")
                    return [data]
                
                elif isinstance(data, list):
                    # Format: [{...}, {...}]
                    # Already an array
                    return data
                
                else:
                    # Unexpected format
                    bt.logging.warning(f"OpenAI response in unexpected format: {data}")
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
    miner = PerplexityMiner()
    miner.run()