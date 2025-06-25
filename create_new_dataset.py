#!/usr/bin/env python3

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedEventProcessor:
    def __init__(self):
        """
        Initialize the enhanced processor with both DeepSeek and OpenAI clients
        """
        # Get API keys from environment variables
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not deepseek_api_key:
            logger.error("DEEPSEEK_API_KEY not found in environment variables")
            sys.exit(1)
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            sys.exit(1)
        
        # Initialize clients
        self.deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        
        self.openai_client = OpenAI(
            api_key=openai_api_key
        )
        
        # Define paths
        self.raw_data_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/raw")
        self.final_output_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/processed/agentB")
        self.checkpoint_dir_a = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/checkpoints/agentA")
        self.checkpoint_dir_b = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/checkpoints/agentB")
        self.logs_dir_a = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/logs/agentA")
        self.logs_dir_b = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/logs/agentB")
        self.prompt_file_a = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/prompts/agentA_prompt.txt")
        self.prompt_file_b = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/prompts/agentB_prompt.txt")
        
        # Create directories if they don't exist
        for directory in [self.final_output_dir, self.checkpoint_dir_a, self.checkpoint_dir_b,
                         self.logs_dir_a, self.logs_dir_b]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load system prompts
        self.system_prompt_a = self.load_system_prompt(self.prompt_file_a, "Agent A")
        self.system_prompt_b = self.load_system_prompt(self.prompt_file_b, "Agent B")
        
    def load_system_prompt(self, prompt_file: Path, agent_name: str) -> str:
        """Load the system prompt from file"""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"{agent_name} system prompt file not found: {prompt_file}")
            logger.error("Please ensure the prompt file exists at the specified path")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading {agent_name} system prompt file: {e}")
            sys.exit(1)

    def validate_json_response_agent_a(self, response_text: str, para_id: str) -> Optional[Dict[str, Any]]:
        """Validate and parse Agent A JSON response - expects format with para_id as key"""[1]
        try:
            response_text = response_text.strip()
            
            # Remove markdown fences
            if response_text.startswith('```'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            parsed = json.loads(response_text)
            
            # Handle different response formats
            if isinstance(parsed, list):
                # If response is a list, wrap it as event_mentions
                logger.info(f"Agent A returned list format for {para_id}, wrapping in event_mentions")
                return {"event_mentions": parsed}
            elif isinstance(parsed, dict):
                # Check if it has the expected format: {para_id: {event_mentions: [...]}}
                if para_id in parsed and isinstance(parsed[para_id], dict) and "event_mentions" in parsed[para_id]:
                    logger.info(f"Agent A returned expected format for {para_id}")
                    return parsed[para_id]
                # Check if it directly has event_mentions
                elif "event_mentions" in parsed:
                    logger.info(f"Agent A returned direct event_mentions format for {para_id}")
                    return parsed
                # Check if any key contains event_mentions structure
                else:
                    for key, value in parsed.items():
                        if isinstance(value, dict) and "event_mentions" in value:
                            logger.info(f"Agent A returned nested format for {para_id}, extracting from key: {key}")
                            return value
                    # No event_mentions found, return empty
                    logger.warning(f"Agent A returned dict without event_mentions for {para_id}")
                    return {"event_mentions": []}
            else:
                logger.warning(f"Agent A returned unexpected type {type(parsed)} for {para_id}")
                return {"event_mentions": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Agent A JSON decode error for {para_id}: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Agent A response validation error for {para_id}: {e}")
            return None

    def validate_json_response_agent_b(self, response_text: str, para_id: str) -> Optional[Dict[str, Any]]:
        """Validate and parse Agent B JSON response - expects format {<id>: {event_mentions: [...]}}"""
        try:
            response_text = response_text.strip()
            
            # Remove markdown fences
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            parsed = json.loads(response_text)
            
            # Handle different response formats for Agent B
            if isinstance(parsed, dict):
                # Check if it has the expected format: {para_id: {event_mentions: [...]}}
                if para_id in parsed and isinstance(parsed[para_id], dict) and "event_mentions" in parsed[para_id]:
                    logger.info(f"Agent B returned expected format for {para_id}")
                    return parsed[para_id]
                # Check if it directly has event_mentions (Agent B sometimes returns this)
                elif "event_mentions" in parsed:
                    logger.info(f"Agent B returned direct event_mentions format for {para_id}")
                    return parsed
                # Check if any key contains event_mentions structure
                else:
                    for key, value in parsed.items():
                        if isinstance(value, dict) and "event_mentions" in value:
                            logger.info(f"Agent B returned nested format for {para_id}, extracting from key: {key}")
                            return value
                    # No event_mentions found, return empty
                    logger.warning(f"Agent B returned dict without event_mentions for {para_id}")
                    return {"event_mentions": []}
            elif isinstance(parsed, list):
                # If response is a list, wrap it as event_mentions
                logger.info(f"Agent B returned list format for {para_id}, wrapping in event_mentions")
                return {"event_mentions": parsed}
            else:
                logger.warning(f"Agent B returned unexpected type {type(parsed)} for {para_id}")
                return {"event_mentions": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Agent B JSON decode error for {para_id}: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Agent B response validation error for {para_id}: {e}")
            return None

    def save_batch_api_log(self, agent: str, batch_filename: str, batch_log_data: Dict):
        """Save batch-level API logs"""
        log_dir = self.logs_dir_a if agent == "A" else self.logs_dir_b
        log_file = log_dir / f"{batch_filename}_batch_log.json"
        
        try:
            # Load existing batch log if it exists
            existing_log = {}
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            
            # Update with new data
            existing_log.update(batch_log_data)
            
            # Save updated log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {agent} batch log for {batch_filename} with {len(batch_log_data)} entries")
        except Exception as e:
            logger.error(f"Failed to save {agent} batch log for {batch_filename}: {e}")

    def call_deepseek_api(self, user_message: str, para_id: str, max_retries: int = 3) -> tuple[Optional[str], Dict]:
        """Call DeepSeek API with retry logic and logging"""
        request_data = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": self.system_prompt_a},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "stream": False,
        }
        
        for attempt in range(max_retries):
            try:
                response = self.deepseek_client.chat.completions.create(**request_data)
                
                # Extract response data
                response_text = response.choices[0].message.content.strip()
                
                # Extract reasoning if available
                reasoning_text = getattr(response.choices[0].message, 'reasoning', None)
                if reasoning_text:
                    reasoning_text = reasoning_text.strip()
                
                # Extract token usage
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                
                # Return detailed log data for batch logging
                api_log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "para_id": para_id,
                    "request": request_data,
                    "response": {"content": response_text},
                    "tokens": tokens_used,
                    "reasoning_text": reasoning_text
                }
                
                logger.info(f"DeepSeek API call successful for {para_id}, tokens: {tokens_used['total_tokens']}")
                return response_text, api_log_data
                
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(f"DeepSeek rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif "timeout" in error_str.lower():
                    logger.warning(f"DeepSeek request timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                else:
                    logger.error(f"DeepSeek API call failed on attempt {attempt + 1}: {error_str}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
        
        return None, {}

    def call_openai_api(self, input_object: Dict, para_id: str, max_retries: int = 3) -> tuple[Optional[str], Dict]:
        """Call OpenAI O4-Mini API with retry logic and logging"""
        user_message = json.dumps(input_object, ensure_ascii=False, indent=2)
        
        request_data = {
            "model": "o4-mini-2025-04-16",
            "messages": [
                {"role": "system", "content": self.system_prompt_b},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
        }
        
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(**request_data)
                
                # Extract response data
                response_text = response.choices[0].message.content.strip()
                
                # Extract reasoning if available
                reasoning_text = getattr(response.choices[0].message, 'reasoning', None)
                if reasoning_text:
                    reasoning_text = reasoning_text.strip()
                
                # Extract token usage
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                
                # Return detailed log data for batch logging
                api_log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "para_id": para_id,
                    "request": request_data,
                    "response": {"content": response_text},
                    "tokens": tokens_used,
                    "reasoning_text": reasoning_text
                }
                
                logger.info(f"OpenAI API call successful for {para_id}, tokens: {tokens_used['total_tokens']}")
                return response_text, api_log_data
                
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(f"OpenAI rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif "timeout" in error_str.lower():
                    logger.warning(f"OpenAI request timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                else:
                    logger.error(f"OpenAI API call failed on attempt {attempt + 1}: {error_str}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
        
        return None, {}

    def load_checkpoint(self, batch_filename: str, agent: str) -> Dict[str, Any]:
        """Load checkpoint for a batch file and agent"""
        checkpoint_dir = self.checkpoint_dir_a if agent == "A" else self.checkpoint_dir_b
        checkpoint_file = checkpoint_dir / f"{batch_filename}_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {agent} checkpoint: {e}")
        
        return {
            "batch_file": batch_filename,
            "agent": agent,
            "processed_ids": [],
            "failed_ids": [],
            "total_ids": 0,
            "last_processed_time": None,
            "status": "not_started",
            "agent_a_results": {}  # Store Agent A results for Agent B
        }

    def save_checkpoint(self, batch_filename: str, agent: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint for a batch file and agent"""
        checkpoint_dir = self.checkpoint_dir_a if agent == "A" else self.checkpoint_dir_b
        checkpoint_file = checkpoint_dir / f"{batch_filename}_checkpoint.json"
        checkpoint_data["last_processed_time"] = datetime.now().isoformat()
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {agent} checkpoint for {batch_filename}")
        except Exception as e:
            logger.error(f"Failed to save {agent} checkpoint: {e}")

    def save_final_results(self, batch_filename: str, results: Dict[str, Any]):
        """Save final Agent B results to the output directory"""
        output_file = self.final_output_dir / f"{batch_filename}_final.json"
        
        try:
            # Load existing results if file exists
            existing_results = {}
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            
            # Merge new results
            existing_results.update(results)
            
            # Save updated results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved final results for {batch_filename} with {len(results)} new entries")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

    def process_single_paradoc_agent_a(self, para_id: str, para_doc: List[str], batch_filename: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict]]:
        """Process a single paradoc entry with Agent A (DeepSeek)"""
        # Create input format expected by the model
        input_data = {para_id: {"para_doc": para_doc}}
        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
        
        logger.info(f"Agent A processing {para_id}...")
        
        # Call DeepSeek API
        response, api_log_data = self.call_deepseek_api(input_json, para_id)
        if response is None:
            logger.error(f"Agent A failed to get response for {para_id}")
            return None, None
        
        # Validate and parse response
        result = self.validate_json_response_agent_a(response, para_id)
        if result is None:
            logger.error(f"Agent A failed to parse response for {para_id}")
            logger.error(f"Raw response: {response[:200]}...")
            return None, api_log_data
        
        # Check if result has event_mentions and is not empty
        if "event_mentions" in result and len(result["event_mentions"]) > 0:
            logger.info(f"Agent A successfully processed {para_id} with {len(result['event_mentions'])} events")
        else:
            logger.info(f"Agent A processed {para_id} but found no events")
        
        return result, api_log_data

    def process_single_paradoc_agent_b(self, para_id: str, paragraph: str, draft_json: Dict, batch_filename: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict]]:
        """Process a single paradoc entry with Agent B (OpenAI O4-Mini)"""
        # Create input object for Agent B
        input_object = {
            "id": para_id,
            "paragraph": paragraph,
            "draft_json": draft_json
        }
        
        logger.info(f"Agent B reviewing {para_id}...")
        
        # Call OpenAI API
        response, api_log_data = self.call_openai_api(input_object, para_id)
        if response is None:
            logger.error(f"Agent B failed to get response for {para_id}")
            return None, None
        
        # Validate and parse response
        result = self.validate_json_response_agent_b(response, para_id)
        if result is None:
            logger.error(f"Agent B failed to parse response for {para_id}")
            logger.error(f"Raw response: {response[:200]}...")
            return None, api_log_data
        
        # Check if result has event_mentions
        if "event_mentions" in result:
            logger.info(f"Agent B successfully reviewed {para_id} with {len(result['event_mentions'])} events")
        else:
            logger.info(f"Agent B reviewed {para_id} but result has no event_mentions")
        
        return result, api_log_data

    def process_batch_file(self, batch_filename: str):
        """Process a single batch file with both agents"""
        batch_file_path = self.raw_data_dir / batch_filename
        
        if not batch_file_path.exists():
            logger.error(f"Batch file not found: {batch_file_path}")
            return
        
        logger.info(f"Starting to process batch file: {batch_filename}")
        
        # Load checkpoints for both agents
        checkpoint_a = self.load_checkpoint(batch_filename, "A")
        checkpoint_b = self.load_checkpoint(batch_filename, "B")
        
        processed_ids_a = set(checkpoint_a.get("processed_ids", []))
        failed_ids_a = set(checkpoint_a.get("failed_ids", []))
        processed_ids_b = set(checkpoint_b.get("processed_ids", []))
        failed_ids_b = set(checkpoint_b.get("failed_ids", []))
        
        # Load batch data
        try:
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load batch file: {e}")
            return
        
        logger.info(f"Loaded batch file with {len(batch_data)} items")
        
        # Update checkpoints with total count
        checkpoint_a["total_ids"] = len(batch_data)
        checkpoint_a["status"] = "processing"
        checkpoint_b["total_ids"] = len(batch_data)
        checkpoint_b["status"] = "processing"
        
        # Initialize batch logs
        batch_log_a = {}
        batch_log_b = {}
        
        # Phase 1: Process all items with Agent A
        logger.info(f"Phase 1: Processing {len(batch_data)} items with Agent A...")
        
        for para_id, para_info in batch_data.items():
            para_doc = para_info.get("para_doc", [])
            
            if para_id not in processed_ids_a and para_id not in failed_ids_a:
                try:
                    result_a, api_log_data = self.process_single_paradoc_agent_a(para_id, para_doc, batch_filename)
                    
                    if result_a is not None:
                        # Store result in checkpoint for Agent B
                        checkpoint_a.setdefault("agent_a_results", {})[para_id] = result_a
                        
                        # Store API log for batch logging
                        if api_log_data:
                            batch_log_a[para_id] = api_log_data
                        
                        # Update checkpoint
                        processed_ids_a.add(para_id)
                        checkpoint_a["processed_ids"] = list(processed_ids_a)
                        
                        logger.info(f"Agent A successfully processed {para_id}")
                    else:
                        # Handle failure
                        failed_ids_a.add(para_id)
                        checkpoint_a["failed_ids"] = list(failed_ids_a)
                        logger.error(f"Agent A failed to process {para_id}")
                    
                    # Save checkpoint after each item
                    self.save_checkpoint(batch_filename, "A", checkpoint_a)
                    
                except Exception as e:
                    logger.error(f"Agent A error processing {para_id}: {e}")
                    failed_ids_a.add(para_id)
                    checkpoint_a["failed_ids"] = list(failed_ids_a)
                    self.save_checkpoint(batch_filename, "A", checkpoint_a)
                    continue
            else:
                logger.info(f"Agent A skipping {para_id} (already processed or failed)")
            
            # Add small delay to avoid overwhelming the API
            time.sleep(1.0)
        
        # Save Agent A batch logs
        if batch_log_a:
            self.save_batch_api_log("A", batch_filename, batch_log_a)
        
        # Mark Agent A as completed
        checkpoint_a["status"] = "completed"
        self.save_checkpoint(batch_filename, "A", checkpoint_a)
        
        logger.info(f"Agent A completed. Processed: {len(processed_ids_a)}, Failed: {len(failed_ids_a)}")
        
        # Phase 2: Process all successful Agent A results with Agent B
        logger.info(f"Phase 2: Processing {len(processed_ids_a)} items with Agent B...")
        
        # Load Agent A results from checkpoint
        agent_a_results = checkpoint_a.get("agent_a_results", {})
        
        # Store Agent A results in Agent B checkpoint for reference
        checkpoint_b["agent_a_results"] = agent_a_results
        
        final_results = {}
        
        for para_id in processed_ids_a:
            if para_id not in processed_ids_b and para_id not in failed_ids_b:
                if para_id in agent_a_results:
                    para_info = batch_data[para_id]
                    para_doc = para_info.get("para_doc", [])
                    paragraph_text = " ".join(para_doc)
                    draft_json = agent_a_results[para_id]
                    
                    # Only process if Agent A found events or if we want to review empty results too
                    if "event_mentions" in draft_json and (len(draft_json["event_mentions"]) > 0 or True):  # Set to True to review all
                        try:
                            result_b, api_log_data = self.process_single_paradoc_agent_b(
                                para_id, paragraph_text, draft_json, batch_filename
                            )
                            
                            if result_b is not None:
                                # Store final result
                                final_results[para_id] = result_b
                                
                                # Store API log for batch logging
                                if api_log_data:
                                    batch_log_b[para_id] = api_log_data
                                
                                # Update checkpoint
                                processed_ids_b.add(para_id)
                                checkpoint_b["processed_ids"] = list(processed_ids_b)
                                
                                logger.info(f"Agent B successfully processed {para_id}")
                            else:
                                # Handle failure
                                failed_ids_b.add(para_id)
                                checkpoint_b["failed_ids"] = list(failed_ids_b)
                                logger.error(f"Agent B failed to process {para_id}")
                            
                            # Save checkpoint after each item
                            self.save_checkpoint(batch_filename, "B", checkpoint_b)
                            
                        except Exception as e:
                            logger.error(f"Agent B error processing {para_id}: {e}")
                            failed_ids_b.add(para_id)
                            checkpoint_b["failed_ids"] = list(failed_ids_b)
                            self.save_checkpoint(batch_filename, "B", checkpoint_b)
                            continue
                    else:
                        logger.info(f"Skipping Agent B for {para_id} - no events found by Agent A")
                        # Still mark as processed
                        processed_ids_b.add(para_id)
                        checkpoint_b["processed_ids"] = list(processed_ids_b)
                        final_results[para_id] = {"event_mentions": []}
                else:
                    logger.warning(f"Agent A result not found for {para_id}")
                    continue
            else:
                logger.info(f"Agent B skipping {para_id} (already processed or failed)")
            
            # Save final results periodically (every 10 items)
            if len(final_results) >= 10:
                self.save_final_results(batch_filename, final_results)
                final_results = {}
            
            # Add small delay to avoid overwhelming the API
            time.sleep(1.0)
        
        # Save any remaining final results
        if final_results:
            self.save_final_results(batch_filename, final_results)
        
        # Save Agent B batch logs
        if batch_log_b:
            self.save_batch_api_log("B", batch_filename, batch_log_b)
        
        # Mark Agent B as completed
        checkpoint_b["status"] = "completed"
        self.save_checkpoint(batch_filename, "B", checkpoint_b)
        
        logger.info(f"Completed processing batch file: {batch_filename}")
        logger.info(f"Agent A - Processed: {len(processed_ids_a)}, Failed: {len(failed_ids_a)}")
        logger.info(f"Agent B - Processed: {len(processed_ids_b)}, Failed: {len(failed_ids_b)}")

    def process_all_files(self):
        """Process all batch files in the raw data directory"""
        batch_files = sorted([f for f in os.listdir(self.raw_data_dir) 
                             if f.startswith('tokenized_data_') and f.endswith('.json')])
        
        logger.info(f"Found {len(batch_files)} batch files to process")
        
        for batch_file in batch_files:
            try:
                self.process_batch_file(batch_file)
            except Exception as e:
                logger.error(f"Error processing batch file {batch_file}: {e}")
                continue

    def get_processing_status(self):
        """Get processing status for all batch files and both agents"""
        batch_files = sorted([f for f in os.listdir(self.raw_data_dir) 
                             if f.startswith('tokenized_data_') and f.endswith('.json')])
        
        total_processed_a = 0
        total_failed_a = 0
        total_processed_b = 0
        total_failed_b = 0
        total_items = 0
        
        print("=" * 80)
        print("PROCESSING STATUS")
        print("=" * 80)
        
        for batch_file in batch_files:
            checkpoint_a = self.load_checkpoint(batch_file, "A")
            checkpoint_b = self.load_checkpoint(batch_file, "B")
            
            processed_count_a = len(checkpoint_a.get("processed_ids", []))
            failed_count_a = len(checkpoint_a.get("failed_ids", []))
            processed_count_b = len(checkpoint_b.get("processed_ids", []))
            failed_count_b = len(checkpoint_b.get("failed_ids", []))
            total_count = checkpoint_a.get("total_ids", 0)
            
            status_a = checkpoint_a.get("status", "not_started")
            status_b = checkpoint_b.get("status", "not_started")
            
            total_processed_a += processed_count_a
            total_failed_a += failed_count_a
            total_processed_b += processed_count_b
            total_failed_b += failed_count_b
            total_items += total_count
            
            print(f"\n{batch_file}:")
            print(f"  Agent A: {processed_count_a} processed, {failed_count_a} failed ({status_a})")
            print(f"  Agent B: {processed_count_b} processed, {failed_count_b} failed ({status_b})")
            print(f"  Total: {total_count}")
        
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Agent A: {total_processed_a} processed, {total_failed_a} failed")
        print(f"Agent B: {total_processed_b} processed, {total_failed_b} failed")
        print(f"Total items: {total_items}")
        
        if total_items > 0:
            success_rate_a = (total_processed_a / total_items) * 100
            success_rate_b = (total_processed_b / total_items) * 100
            print(f"Agent A success rate: {success_rate_a:.2f}%")
            print(f"Agent B success rate: {success_rate_b:.2f}%")

def main():
    """Main function"""
    # Initialize processor
    processor = EnhancedEventProcessor()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            # Show processing status
            processor.get_processing_status()
        else:
            # Process specific file
            batch_filename = sys.argv[1]
            if not batch_filename.endswith('.json'):
                batch_filename += '.json'
            processor.process_batch_file(batch_filename)
    else:
        # Process all files
        processor.process_all_files()

if __name__ == "__main__":
    main()
