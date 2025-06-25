#!/usr/bin/env python3
"""
Agent A: DeepSeek Processor with Thread-Safe Parallel Processing
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
import concurrent.futures
import multiprocessing
import threading
from dotenv import load_dotenv

load_dotenv()

# Configure logging with thread safety
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agentA/agentA_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepSeekProcessor:
    def __init__(self):
        self.raw_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/raw")
        self.output_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/processed/agentA")
        self.checkpoint_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/checkpoints/agentA")
        self.logs_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/logs/agentA")

        # Initialize directories
        for d in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Check API key
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        if not deepseek_api_key:
            logger.error("DEEPSEEK_API_KEY not found in environment variables")
            sys.exit(1)

        self.api_key = deepseek_api_key  # Store key, create clients per thread
        
        # Load system prompt
        self.prompt_file = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/prompts/agentA_prompt.txt")
        self.system_prompt = self.load_system_prompt(self.prompt_file)
        
        # THREAD SAFETY: Add locks for shared resources
        self.results_lock = threading.Lock()
        self.checkpoint_lock = threading.Lock()
        self.log_lock = threading.Lock()

    def create_client(self):
        """Create a new client for each thread"""
        return OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def load_system_prompt(self, prompt_file: Path) -> str:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            sys.exit(1)

    def log_complete_interaction(self, para_id: str, input_data: Dict, response_content: str, 
                               reasoning_content: str, tokens_used: Dict, success: bool, error_msg: str = None):
        """Thread-safe logging of complete interaction"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "para_id": para_id,
            "thread_name": threading.current_thread().name,
            "success": success,
            "input": {
                "system_prompt": self.system_prompt,
                "user_message": input_data.get("user_message", ""),
                "request_data": input_data.get("request_data", {})
            },
            "output": {
                "content": response_content,
                "reasoning_content": reasoning_content,
                "tokens": tokens_used
            },
            "error": error_msg
        }
        
        log_file = self.logs_dir / f"{para_id}_complete_interaction.json"
        with self.log_lock:  # THREAD SAFETY
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_entry, f, ensure_ascii=False, indent=2)
                logger.info(f"[{threading.current_thread().name}] Saved complete interaction log for {para_id}")
            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Failed to save interaction log for {para_id}: {e}")
    def fix_nested_event_mentions(self, batch_filename: str):
        """Fix existing files with nested event_mentions structure"""
        output_file = self.output_dir / f"{batch_filename}_agentA.json"
        
        if not output_file.exists():
            return
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            fixed_data = {}
            for para_id, content in data.items():
                if "event_mentions" in content:
                    new_events = []
                    for event in content["event_mentions"]:
                        # Check if event is nested structure
                        if isinstance(event, dict) and para_id in event and "event_mentions" in event[para_id]:
                            # Unwrap the nested event_mentions
                            nested_events = event[para_id]["event_mentions"]
                            new_events.extend(nested_events)
                        else:
                            new_events.append(event)
                    fixed_data[para_id] = {"event_mentions": new_events}
                else:
                    fixed_data[para_id] = content
            
            # Save fixed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(fixed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Fixed nested structure in {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to fix nested structure: {e}")
        
    def validate_json_response(self, response_text: str, para_id: str) -> Optional[Dict[str, Any]]:
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
            
            # Handle list format
            if isinstance(parsed, list):
                logger.info(f"[{threading.current_thread().name}] Agent A returned list format for {para_id}")
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    first_item = parsed[0]  # ✅ Get first item
                    if para_id in first_item and "event_mentions" in first_item[para_id]:
                        return first_item[para_id]  # ✅ Return just event structure
                    elif "event_mentions" in first_item:
                        return first_item  # ✅ Return first item
                    else:
                        # Check all keys for event_mentions
                        for key, value in first_item.items():
                            if isinstance(value, dict) and "event_mentions" in value:
                                return value  # ✅ Return just the event structure
                        return {"event_mentions": []}  # ✅ Empty if nothing found
                else:
                    return {"event_mentions": []}
            
            # Handle dict format
            elif isinstance(parsed, dict):
                if para_id in parsed and "event_mentions" in parsed[para_id]:
                    return parsed[para_id]
                elif "event_mentions" in parsed:
                    return parsed
                else:
                    for key, value in parsed.items():
                        if isinstance(value, dict) and "event_mentions" in value:
                            return value
                    return {"event_mentions": []}
            else:
                return {"event_mentions": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"[{threading.current_thread().name}] JSON parse error for {para_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"[{threading.current_thread().name}] Response validation error for {para_id}: {e}")
            return None

    def call_deepseek_api(self, user_message: str, para_id: str, max_retries: int = 3) -> tuple[Optional[str], Dict]:
        # CREATE CLIENT PER THREAD
        client = self.create_client()
        
        request_data = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "stream": False,
        }
        
        input_log = {
            "request_data": request_data,
            "user_message": user_message
        }
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**request_data)
                
                content = response.choices[0].message.content.strip()
                reasoning_content = response.choices[0].message.reasoning_content
                if reasoning_content:
                    reasoning_content = reasoning_content.strip()
                else:
                    reasoning_content = ""
                
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                
                self.log_complete_interaction(
                    para_id, input_log, content, reasoning_content, tokens_used, True
                )
                
                response_data = {
                    "content": content,
                    "reasoning": reasoning_content,
                    "tokens": tokens_used,
                    "model": request_data["model"],
                    "attempt": attempt + 1
                }
                
                logger.info(f"[{threading.current_thread().name}] DeepSeek API call successful for {para_id}, tokens: {tokens_used['total_tokens']}")
                return content, response_data
                
            except Exception as e:
                error_str = str(e)
                self.log_complete_interaction(
                    para_id, input_log, "", "", {}, False, error_str
                )
                
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    # ADD JITTER to prevent thundering herd
                    wait_time = (2 ** attempt) + (threading.get_ident() % 5)
                    logger.warning(f"[{threading.current_thread().name}] DeepSeek rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif "timeout" in error_str.lower():
                    logger.warning(f"[{threading.current_thread().name}] DeepSeek request timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 + (threading.get_ident() % 3))
                        continue
                else:
                    logger.error(f"[{threading.current_thread().name}] DeepSeek API call failed on attempt {attempt + 1}: {error_str}")
                    if attempt < max_retries - 1:
                        time.sleep(2 + (threading.get_ident() % 3))
                        continue
        
        return None, {}

    def save_results(self, batch_filename: str, results: Dict[str, Any]):
        """THREAD-SAFE: Save Agent A intermediate results to batch file"""
        output_file = self.output_dir / f"{batch_filename}_agentA.json"
        
        with self.results_lock:  # THREAD SAFETY
            try:
                existing_results = {}
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                
                existing_results.update(results)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"[{threading.current_thread().name}] Successfully saved {len(results)} new entries")
                
            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Failed to save Agent A results: {e}")
                raise

    def save_checkpoint(self, batch_filename: str, checkpoint_data: Dict[str, Any]):
        """THREAD-SAFE: Save checkpoint for a batch file"""
        checkpoint_file = self.checkpoint_dir / f"{batch_filename}_checkpoint.json"
        checkpoint_data["last_processed_time"] = datetime.now().isoformat()
        
        with self.checkpoint_lock:  # THREAD SAFETY
            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                logger.debug(f"[{threading.current_thread().name}] Saved checkpoint for {batch_filename}")
            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Failed to save checkpoint: {e}")

    def load_results(self, batch_filename: str) -> Dict[str, Any]:
        """Load existing Agent A results from batch file"""
        output_file = self.output_dir / f"{batch_filename}_agentA.json"
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Loaded {len(results)} existing Agent A results for {batch_filename}")
                return results
            except Exception as e:
                logger.error(f"Failed to load Agent A results: {e}")
        return {}

    def load_checkpoint(self, batch_filename: str) -> Dict[str, Any]:
        """Load checkpoint for a batch file"""
        checkpoint_file = self.checkpoint_dir / f"{batch_filename}_checkpoint.json"
        checkpoint = {
            "batch_file": batch_filename,
            "processed_ids": [],
            "failed_ids": [],
            "total_ids": 0,
            "last_processed_time": None,
            "status": "not_started"
        }
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    existing_checkpoint = json.load(f)
                    checkpoint.update(existing_checkpoint)
                logger.info(f"Loaded existing checkpoint for {batch_filename}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        
        # Update processed_ids from saved results
        existing_results = self.load_results(batch_filename)
        if existing_results:
            existing_processed_ids = set(existing_results.keys())
            checkpoint_processed_ids = set(checkpoint.get("processed_ids", []))
            all_processed_ids = existing_processed_ids.union(checkpoint_processed_ids)
            checkpoint["processed_ids"] = list(all_processed_ids)
            logger.info(f"Total processed IDs from results and checkpoint: {len(all_processed_ids)}")
        
        return checkpoint

    def process_single_paradoc(self, para_id: str, para_doc: List[str], batch_filename: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict]]:
        """Process a single paradoc entry"""
        input_data = {para_id: {"para_doc": para_doc}}
        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
        
        logger.info(f"[{threading.current_thread().name}] Processing {para_id}...")
        
        response, response_data = self.call_deepseek_api(input_json, para_id)
        if response is None:
            logger.error(f"[{threading.current_thread().name}] Failed to get response for {para_id}")
            return None, None
        
        result = self.validate_json_response(response, para_id)
        if result is None:
            logger.error(f"[{threading.current_thread().name}] Failed to parse response for {para_id}")
            return None, response_data
        
        if "event_mentions" in result and len(result["event_mentions"]) > 0:
            logger.info(f"[{threading.current_thread().name}] Successfully processed {para_id} with {len(result['event_mentions'])} events")
        else:
            logger.info(f"[{threading.current_thread().name}] Processed {para_id} but found no events")
        
        return result, response_data

    def save_batch_summary(self, batch_filename: str, batch_summary: Dict):
        """Save batch-level summary log"""
        log_file = self.logs_dir / f"{batch_filename}_batch_summary.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved batch summary for {batch_filename}")
        except Exception as e:
            logger.error(f"Failed to save batch summary: {e}")

    def process_batch_file(self, batch_filename: str):
        """Process a single batch file, dispatching API calls in parallel."""
        batch_file_path = self.raw_dir / batch_filename
        if not batch_file_path.exists():
            logger.error(f"Batch file not found: {batch_file_path}")
            return

        logger.info(f"Starting to process batch file: {batch_filename}")

        # Load checkpoint
        checkpoint = self.load_checkpoint(batch_filename)
        processed_ids = set(checkpoint.get("processed_ids", []))
        failed_ids = set(checkpoint.get("failed_ids", []))

        # Load batch data
        try:
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load batch file: {e}")
            return

        total_items = len(batch_data)
        logger.info(f"Loaded batch with {total_items} items "
                    f"(already processed: {len(processed_ids)}, failed: {len(failed_ids)})")

        # Update checkpoint status
        checkpoint.update({
            "total_ids": total_items,
            "status": "processing"
        })
        self.save_checkpoint(batch_filename, checkpoint)

        # Prepare summary
        batch_summary = {
            "batch_file": batch_filename,
            "start_time": datetime.now().isoformat(),
            "total_items": total_items,
            "processed_count": 0,
            "failed_count": 0,
            "items": {}
        }

        # Which IDs are still to do
        remaining = [
            pid for pid in batch_data
            if pid not in processed_ids and pid not in failed_ids
        ]
        n_workers = multiprocessing.cpu_count()
        # n_workers = 2
        logger.info(f"Dispatching {len(remaining)} items on {n_workers} threads")

        # Helper that just wraps your single‐doc processor
        def _worker(pid: str):
            try:
                return pid, *self.process_single_paradoc(pid,
                                                         batch_data[pid].get("para_doc", []),
                                                         batch_filename)
            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Worker error for {pid}: {e}")
                return pid, None, None

        # Execute in thread pool with proper error handling
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as execr:
            futures = [execr.submit(_worker, pid) for pid in remaining]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    pid, result, response_data = fut.result()
                    
                    if result is not None:
                        # persist result immediately
                        self.save_results(batch_filename, {pid: result})
                        
                        batch_summary["items"][pid] = {
                            "status": "success",
                            "events_found": len(result.get("event_mentions", [])),
                            "tokens": response_data.get("tokens", {}) if response_data else {},
                            "reasoning_length": len(response_data.get("reasoning", "")) if response_data else 0,
                        }
                        batch_summary["processed_count"] += 1
                        processed_ids.add(pid)
                        checkpoint["processed_ids"] = list(processed_ids)
                        logger.info(f"[{pid}] success ({batch_summary['items'][pid]['events_found']} events)")
                    else:
                        batch_summary["items"][pid] = {
                            "status": "failed",
                            "error": "Failed to process"
                        }
                        batch_summary["failed_count"] += 1
                        failed_ids.add(pid)
                        checkpoint["failed_ids"] = list(failed_ids)
                        logger.error(f"[{pid}] failed to process")

                    # update checkpoint after each completion
                    self.save_checkpoint(batch_filename, checkpoint)
                    
                except Exception as e:
                    logger.error(f"Error processing future result: {e}")

        # Batch is done—finalize summary and checkpoint
        batch_summary["end_time"] = datetime.now().isoformat()
        self.save_batch_summary(batch_filename, batch_summary)

        checkpoint["status"] = "completed"
        self.save_checkpoint(batch_filename, checkpoint)

        logger.info(f"Completed {batch_filename}: "
                    f"{batch_summary['processed_count']} processed, "
                    f"{batch_summary['failed_count']} failed")

    def process_all_files(self):
        """Process all batch files in the raw data directory"""
        batch_files = sorted([f for f in os.listdir(self.raw_dir) 
                             if f.startswith('tokenized_data_') and f.endswith('.json')])
        logger.info(f"Found {len(batch_files)} batch files to process")
        
        for batch_file in batch_files:
            try:
                self.process_batch_file(batch_file)
            except Exception as e:
                logger.error(f"Error processing batch file {batch_file}: {e}")

    def get_processing_status(self):
        """Get processing status for all batch files"""
        batch_files = sorted([f for f in os.listdir(self.raw_dir) 
                             if f.startswith('tokenized_data_') and f.endswith('.json')])
        
        print("=" * 80)
        print("PARALLEL AGENT A PROCESSING STATUS")
        print("=" * 80)
        
        total_processed = 0
        total_failed = 0
        total_items = 0
        
        for batch_file in batch_files:
            checkpoint = self.load_checkpoint(batch_file)
            processed = len(checkpoint.get("processed_ids", []))
            failed = len(checkpoint.get("failed_ids", []))
            total = checkpoint.get("total_ids", 0)
            status = checkpoint.get("status", "not_started")
            
            total_processed += processed
            total_failed += failed
            total_items += total
            
            print(f"{batch_file}:")
            print(f"  Processed: {processed}, Failed: {failed}, Total: {total}")
            print(f"  Status: {status}")
        
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total processed: {total_processed}")
        print(f"Total failed: {total_failed}")
        print(f"Total items: {total_items}")
        print(f"Using {multiprocessing.cpu_count()} CPU cores")
        
        if total_items > 0:
            success_rate = (total_processed / total_items) * 100
            print(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    processor = DeepSeekProcessor()
    if len(sys.argv) > 1:
        if sys.argv == "--status":
            processor.get_processing_status()
        else:
            batch_filename = sys.argv[1]
            if not batch_filename.endswith('.json'):
                batch_filename += '.json'
            processor.process_batch_file(batch_filename)
            # processor.fix_nested_event_mentions(batch_filename)
    else:
        processor.process_all_files()
