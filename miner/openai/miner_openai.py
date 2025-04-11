import os
import time
import argparse
import traceback
import bittensor as bt
import json
from typing import Tuple, List
import logging

from dotenv import load_dotenv

from shared.log_data import LoggerType
from shared.proxy_log_handler import register_proxy_log_handler
from shared.veridex_protocol import VericoreSynapse, SourceEvidence

# OpenAI client
from openai import OpenAI

# debug
bt.logging.set_trace()

load_dotenv()

class MinerOpenAI:
    def __init__(self):
        self.config = self.get_config()
        self.setup_bittensor_objects()
        self.setup_logging()

        # Load OpenAI API key
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
        if not self.openai_api_key or self.openai_api_key.startswith("YOUR_API_KEY_HERE"):
            bt.logging.warning("No OPENAI_API_KEY found. Please set it in your .env file.")
            
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=self.openai_api_key
        )

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--custom", default="my_custom_value", help="Adds a custom value.")
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID.")
        parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
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
                "miner_openai",
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
        Calls OpenAI. Returns a list of (url, snippet) with supporting evidence.
        """
        bt.logging.info(f"{synapse.request_id} | Received Veridex request")
        statement = synapse.statement
        sources = synapse.sources  # Get preferred sources if provided
        
        bt.logging.info(f"{synapse.request_id} | Statement: {statement}")
        if sources:
            bt.logging.info(f"{synapse.request_id} | Preferred sources: {sources}")
            
        bt.logging.info(f"{synapse.request_id} | Calling OpenAI")
        results = self.call_openai(statement, sources)
        bt.logging.info(f"{synapse.request_id} | Received response from OpenAI")
        
        if not results:
            synapse.veridex_response = []
            return synapse

        final_evidence = []
        for item in results:
            url = item.get("url", "").strip()
            snippet = item.get("snippet", "").strip()
            if url and snippet:
                # Store them in SourceEvidence
                ev = SourceEvidence(url=url, excerpt=snippet)
                final_evidence.append(ev)

        synapse.veridex_response = final_evidence
        bt.logging.info(f"{synapse.request_id} | Miner returns {len(final_evidence)} evidence items for statement: '{statement}'.")
        return synapse

    def call_openai(self, statement: str, preferred_sources: List[str] = None) -> List[dict]:
        """
        1) Provide system & user messages.
        2) Parse JSON from the response -> [ {url, snippet}, ... ].
        """
        system_content = """
You are an API that fact checks statements with high accuracy.

Rules:
1. Return the response **only as a JSON array**.
2. The response **must be a valid JSON array**, formatted as:
   ```json
   [{"url": "<source url>", "snippet": "<snippet that directly agrees with or contradicts statement>"}]
   ```
3. Do not include any introductory text, explanations, or additional commentary.
4. Do not add any labels, headers, or markdown formatting—only return the JSON array.
5. Each snippet must be an exact match of text from the source URL.
6. Include diverse sources when possible (scholarly articles, reputable news outlets, government sites).
7. For controversial topics, include evidence from different perspectives.
8. Focus on factual information rather than opinions.
9. Prefer recent sources when temporal relevance matters.

Steps:
1. Find sources / text segments that either contradict or agree with the user provided statement.
2. Pick and extract the segments that most strongly agree or contradict the statement.
3. Do not return urls or segments that do not directly support or disagree with the statement.
4. Do not change any text in the segments (must return an exact html text match), but do shorten the segment to get only the part that directly agrees or disagrees with the statement.
5. Create the json object for each source and statement and add them only INTO ONE array.

Response MUST be returned as a json array ONLY.
"""
        
        # Modify user content based on whether preferred sources are provided
        if preferred_sources and len(preferred_sources) > 0:
            source_list = ", ".join(preferred_sources)
            user_content = f"""Return snippets that strongly agree with or reject the following statement:
"{statement}"

Prioritize these sources if possible: {source_list}"""
        else:
            user_content = f"""Return snippets that strongly agree with or reject the following statement:
"{statement}"

Find the most reliable and relevant sources available."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw_text = None
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.model,  # Use model from config (default: gpt-4o)
                messages=messages,
                temperature=0.2,  # Lower temperature for more factual responses
                stream=False
            )
            if not hasattr(response, "choices") or len(response.choices) == 0:
                bt.logging.warn(f"OpenAI returned no choices: {response}")
                return []
                
            raw_text = response.choices[0].message.content.strip()
            # Check if the response starts with ```json and ends with ```
            if raw_text.startswith("```json") and raw_text.endswith("```"):
                # Extract JSON part between the backticks
                raw_text = raw_text[7:-3].strip()
                
            data = json.loads(raw_text)
            if not isinstance(data, list):
                bt.logging.warn(f"OpenAI response is not a list: {data}")
                return []
                
            return data
        except Exception as e:
            if raw_text is not None:
                bt.logging.error(f"Raw Text of AI Response: {raw_text}")

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
                bt.logging.success("OpenAI Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    miner = MinerOpenAI()
    miner.run()