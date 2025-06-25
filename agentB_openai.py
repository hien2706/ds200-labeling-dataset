#!/usr/bin/env python3
"""
Agent B: OpenAI O4-Mini Reviewer
Review Agent A's results and produce final output
"""

import json
import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
import multiprocessing as mp

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agentB/agentB_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentBProcessor:
    def __init__(self, max_workers=None):
        self.agent_a_sorted_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/processed/agentA/sorted")
        self.raw_data_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/raw")
        self.output_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/processed/agentB")
        self.checkpoint_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/checkpoints/agentB")
        self.logs_dir = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/logs/agentB")
        self.prompt_file = Path("/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/prompts/agentB_prompt.txt")

        # Initialize directories
        for d in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Check API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            sys.exit(1)

        self.api_key = openai_api_key
        
        # Set max workers
        # self.max_workers = max_workers or mp.cpu_count()
        self.max_workers = 2
        logger.info(f"Using {self.max_workers} worker threads")

        # Load system prompt
        self.system_prompt = self.load_system_prompt(self.prompt_file)
        
        # Thread-safe locks
        self.checkpoint_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.log_lock = threading.Lock()

    def create_client(self):
        """Create a new client instance for each thread"""
        return OpenAI(api_key=self.api_key)

    def load_system_prompt(self, prompt_file: Path) -> str:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            sys.exit(1)

    def log_complete_interaction(self, para_id: str, input_data: Dict, response_content: str, 
                               tokens_used: Dict, success: bool, error_msg: str = None):
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
                "tokens": tokens_used
            },
            "error": error_msg
        }
        
        log_file = self.logs_dir / f"{para_id}_complete_interaction.json"
        with self.log_lock:
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_entry, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to save interaction log for {para_id}: {e}")

    def validate_json_response(self, response_text: str, para_id: str) -> Dict[str, Any]:
        """Validate and parse Agent B JSON response"""
        # Clean the response text
        response_content = response_text.replace('```json', '').replace('```', '').strip()
        
        try:
            response_dict = json.loads(response_content)
            
            # Case 1: Response has para_id as top-level key
            if para_id in response_dict:
                event_data = response_dict[para_id]
                if isinstance(event_data, dict) and "event_mentions" in event_data:
                    return event_data  # Return just the event_mentions structure
            
            # Case 2: Response has event_mentions at top level
            if isinstance(response_dict, dict) and "event_mentions" in response_dict:
                return response_dict
            
            # Case 3: Response is nested, find event_mentions
            if isinstance(response_dict, dict):
                for key, value in response_dict.items():
                    if isinstance(value, dict) and "event_mentions" in value:
                        return value
            
            # Case 4: Response is a list, wrap it
            if isinstance(response_dict, list):
                return {"event_mentions": response_dict}
            
            # Default fallback
            logger.warning(f"Unexpected response structure for {para_id}: {response_dict}")
            return {"event_mentions": []}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response for {para_id}: {response_content}")
            raise e

    def call_openai_api(self, input_object: Dict, para_id: str, max_retries: int = 3) -> tuple[Optional[str], Dict]:
        """Thread-safe OpenAI API call"""
        client = self.create_client()
        
        user_message = json.dumps(input_object, ensure_ascii=False, indent=2)
        
        request_data = {
            "model": "o4-mini-2025-04-16",
            "messages": [
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            "service_tier" : "flex",
        }
        
        input_log = {
            "request_data": request_data,
            "user_message": user_message
        }
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**request_data)
                
                content = response.choices[0].message.content.strip()
                
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                
                self.log_complete_interaction(
                    para_id, input_log, content, tokens_used, True
                )
                
                response_data = {
                    "content": content,
                    "tokens": tokens_used,
                    "model": request_data["model"],
                    "attempt": attempt + 1
                }
                
                logger.info(f"[{threading.current_thread().name}] OpenAI API call successful for {para_id}, tokens: {tokens_used['total_tokens']}")
                return content, response_data
                
            except Exception as e:
                error_str = str(e)
                self.log_complete_interaction(
                    para_id, input_log, "", {}, False, error_str
                )
                
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = (2 ** attempt) + (threading.get_ident() % 5)
                    logger.warning(f"[{threading.current_thread().name}] OpenAI rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif "timeout" in error_str.lower():
                    logger.warning(f"[{threading.current_thread().name}] OpenAI request timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 + (threading.get_ident() % 3))
                        continue
                else:
                    logger.error(f"[{threading.current_thread().name}] OpenAI API call failed on attempt {attempt + 1}: {error_str}")
                    if attempt < max_retries - 1:
                        time.sleep(2 + (threading.get_ident() % 3))
                        continue
        
        return None, {}

    def save_results_thread_safe(self, batch_filename: str, results: Dict[str, Any]):
        """Thread-safe results saving"""
        # Extract the base name without _agentA.json suffix
        base_name = batch_filename.replace("_agentA.json", "")
        output_file = self.output_dir / f"{base_name}_final.json"
        
        with self.results_lock:
            try:
                existing_results = {}
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                
                existing_results.update(results)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"[{threading.current_thread().name}] Saved {len(results)} results to {output_file}")
                
            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Failed to save results: {e}")

    def update_checkpoint_thread_safe(self, batch_filename: str, para_id: str, success: bool):
        """Thread-safe checkpoint updating - only track failed IDs"""
        checkpoint_file = self.checkpoint_dir / f"{batch_filename}_checkpoint.json"
        
        with self.checkpoint_lock:
            try:
                checkpoint = {}
                if checkpoint_file.exists():
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint = json.load(f)
                
                if success:
                    # Remove from failed IDs if it was previously failed
                    failed_ids = set(checkpoint.get("failed_ids", []))
                    failed_ids.discard(para_id)  # Remove if exists
                    checkpoint["failed_ids"] = list(failed_ids)
                    # Don't track processed_ids here - they're in the result file
                else:
                    # Add to failed IDs
                    failed_ids = set(checkpoint.get("failed_ids", []))
                    failed_ids.add(para_id)
                    checkpoint["failed_ids"] = list(failed_ids)
                
                checkpoint["last_processed_time"] = datetime.now().isoformat()
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Failed to update checkpoint: {e}")


    def process_single_item(self, item_data: tuple) -> Dict[str, Any]:
        """Process a single item - designed for parallel execution"""
        para_id, draft_json, paragraph, batch_filename = item_data
        
        try:
            # Create input object for Agent B
            input_object = {
                "id": para_id,
                "paragraph": paragraph,
                "draft_json": draft_json
            }
            
            logger.info(f"[{threading.current_thread().name}] Processing {para_id}...")
            
            # Call OpenAI API
            response, response_data = self.call_openai_api(input_object, para_id)
            if response is None:
                logger.error(f"[{threading.current_thread().name}] Failed to get response for {para_id}")
                self.update_checkpoint_thread_safe(batch_filename, para_id, False)
                return {"para_id": para_id, "success": False, "error": "API call failed"}
            
            # Validate response
            result = self.validate_json_response(response, para_id)
            if result is None:
                logger.error(f"[{threading.current_thread().name}] Failed to parse response for {para_id}")
                self.update_checkpoint_thread_safe(batch_filename, para_id, False)
                return {"para_id": para_id, "success": False, "error": "JSON validation failed"}
            
            # Save result immediately
            self.save_results_thread_safe(batch_filename, {para_id: result})
            self.update_checkpoint_thread_safe(batch_filename, para_id, True)
            
            event_count = len(result.get("event_mentions", []))
            logger.info(f"[{threading.current_thread().name}] Successfully processed {para_id} with {event_count} events")
            
            return {
                "para_id": para_id, 
                "success": True, 
                "events_found": event_count,
                "tokens": response_data.get("tokens", {}) if response_data else {}
            }
            
        except Exception as e:
            logger.error(f"[{threading.current_thread().name}] Error processing {para_id}: {e}")
            self.update_checkpoint_thread_safe(batch_filename, para_id, False)
            return {"para_id": para_id, "success": False, "error": str(e)}

    def load_processed_ids_from_results(self, batch_filename: str) -> set:
        """Load already processed IDs from the result file"""
        base_name = batch_filename.replace("_agentA.json", "")
        output_file = self.output_dir / f"{base_name}_final.json"
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                processed_ids = set(results.keys())
                logger.info(f"Loaded {len(processed_ids)} processed IDs from result file: {output_file}")
                return processed_ids
            except Exception as e:
                logger.error(f"Failed to load result file {output_file}: {e}")
        
        return set()

    def load_checkpoint(self, batch_filename: str) -> Dict[str, Any]:
        """Load checkpoint for a batch file - only track failed IDs, get processed from results"""
        checkpoint_file = self.checkpoint_dir / f"{batch_filename}_checkpoint.json"
        checkpoint = {
            "batch_file": batch_filename,
            "processed_ids": [],  # Will be populated from result file
            "failed_ids": [],
            "total_ids": 0,
            "last_processed_time": None,
            "status": "not_started"
        }
        
        with self.checkpoint_lock:
            # Load checkpoint (mainly for failed IDs)
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        existing_checkpoint = json.load(f)
                        checkpoint.update(existing_checkpoint)
                    logger.info(f"Loaded existing checkpoint for {batch_filename}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}")
            
            # Load processed IDs from result file (this is the source of truth)
            processed_ids_from_results = self.load_processed_ids_from_results(batch_filename)
            checkpoint["processed_ids"] = list(processed_ids_from_results)
            
            logger.info(f"Total processed IDs from result file: {len(processed_ids_from_results)}")
            logger.info(f"Failed IDs from checkpoint: {len(checkpoint.get('failed_ids', []))}")
        
        return checkpoint


    def save_checkpoint(self, batch_filename: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint for a batch file"""
        checkpoint_file = self.checkpoint_dir / f"{batch_filename}_checkpoint.json"
        checkpoint_data["last_processed_time"] = datetime.now().isoformat()
        with self.checkpoint_lock:
            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved checkpoint for {batch_filename}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    def process_batch_file_parallel(self, batch_filename: str):
        """Process batch file using parallel workers"""
        # Paths
        agent_a_file = self.agent_a_sorted_dir / batch_filename
        base_name = batch_filename.replace("_agentA.json", "").replace(".json", "") + ".json"
        raw_file = self.raw_data_dir / base_name
        
        if not agent_a_file.exists():
            logger.error(f"Agent A file not found: {agent_a_file}")
            return
        
        if not raw_file.exists():
            logger.error(f"Raw data file not found: {raw_file}")
            return
        
        logger.info(f"Starting parallel processing of batch file: {batch_filename}")
        
        # Load data and checkpoint
        checkpoint = self.load_checkpoint(batch_filename)
        processed_ids = set(checkpoint.get("processed_ids", []))  # From result file
        failed_ids = set(checkpoint.get("failed_ids", []))        # From checkpoint file
        
        try:
            # Load Agent A results
            with open(agent_a_file, 'r', encoding='utf-8') as f:
                agent_a_data = json.load(f)
            
            # Load raw data for paragraphs
            with open(raw_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input files: {e}")
            return

        # Prepare work items - exclude both processed and failed
        work_items = []
        for para_id, draft_json in agent_a_data.items():
            if para_id not in processed_ids and para_id not in failed_ids:
                # Get original paragraph text
                para_doc = raw_data.get(para_id, {}).get("para_doc", [])
                print(f"paradoc: {para_doc}")
                paragraph = " ".join(para_doc)
                print(f"paragraph: {paragraph}")
                
                # if paragraph:
                #     work_items.append((para_id, draft_json, paragraph, batch_filename))
                # else:
                #     logger.warning(f"No paragraph text found for {para_id}")
                if not paragraph:
                    logger.warning(f"No paragraph text found for {para_id}")
                work_items.append((para_id, draft_json, paragraph, batch_filename))

        logger.info(f"Processing {len(work_items)} items using {self.max_workers} workers")
        logger.info(f"Already processed: {len(processed_ids)}, failed: {len(failed_ids)}")
        logger.info(f"Total in Agent A file: {len(agent_a_data)}")

        if len(work_items) == 0:
            logger.info("No items to process, marking as completed and exiting")
            checkpoint["status"] = "completed"
            self.save_checkpoint(batch_filename, checkpoint)
            return  # Exit early, don't create ThreadPoolExecutor
    
        # Update checkpoint
        checkpoint["total_ids"] = len(agent_a_data)
        checkpoint["status"] = "processing"
    
        
        # Process in parallel
        start_time = time.time()
        successful_count = 0
        failed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_single_item, item): item 
                for item in work_items
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result["success"]:
                        successful_count += 1
                    else:
                        failed_count += 1
                        logger.error(f"Failed to process {result['para_id']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception processing {item}: {e}")

        # Final checkpoint update
        with self.checkpoint_lock:
            checkpoint["status"] = "completed"
            self.save_checkpoint(batch_filename, checkpoint)

        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Completed parallel processing of {batch_filename}")
        logger.info(f"Processed: {successful_count}, Failed: {failed_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        if len(work_items) > 0:
            logger.info(f"Average time per item: {total_time/len(work_items):.2f} seconds")

    def process_all_files(self):
        """Process all Agent A sorted files"""
        batch_files = sorted([f.name for f in self.agent_a_sorted_dir.iterdir() 
                             if f.name.endswith("_agentA.json")])
        
        logger.info(f"Found {len(batch_files)} Agent A sorted files to process")
        
        for batch_file in batch_files:
            try:
                self.process_batch_file_parallel(batch_file)
            except Exception as e:
                logger.error(f"Error processing batch file {batch_file}: {e}")

    def get_processing_status(self):
        """Get processing status for all batch files"""
        batch_files = sorted([f.name for f in self.agent_a_sorted_dir.iterdir() 
                             if f.name.endswith("_agentA.json")])
        
        print("=" * 80)
        print("AGENT B PROCESSING STATUS")
        print("=" * 80)
        
        total_processed = 0
        total_failed = 0
        total_items = 0
        
        for batch_file in batch_files:
            with self.checkpoint_lock:
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
        print(f"Using {self.max_workers} worker threads")
        
        if total_items > 0:
            success_rate = (total_processed / total_items) * 100
            print(f"Success rate: {success_rate:.2f}%")

def main():
    """Main function"""
    processor = AgentBProcessor()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            processor.get_processing_status()
        else:
            batch_filename = sys.argv[1]
            if not batch_filename.endswith("_agentA.json"):
                batch_filename += "_agentA.json"
            processor.process_batch_file_parallel(batch_filename)
    else:
        processor.process_all_files()

if __name__ == "__main__":
    main()
