import asyncio
import logging
from queue import Queue, Empty
from threading import Thread
from typing import List, Dict
from services.memory_extraction_service import MemoryExtractionService

logger = logging.getLogger(__name__)

class MemoryConsolidationQueue:
    """
    Background queue for memory extraction
    Prevents blocking user requests
    """

    def __init__(self, extraction_service: MemoryExtractionService):
        self.extraction_service = extraction_service
        self.queue = Queue()
        self.worker_thread = None
        self.running = False
        logger.info("MemoryConsolidationQueue initialized")

    def start(self):
        """Start background worker"""
        if self.running:
            logger.warning("Consolidation worker already running")
            return

        self.running = True
        self.worker_thread = Thread(target=self._worker, daemon=True, name="MemoryConsolidation")
        self.worker_thread.start()
        logger.info("Memory consolidation worker started")

    def stop(self):
        """Stop background worker"""
        if not self.running:
            return

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Memory consolidation worker stopped")

    def enqueue(self, user_id: int, conversation: List[Dict]):
        """Add conversation to processing queue"""
        self.queue.put({
            'user_id': user_id,
            'conversation': conversation
        })
        logger.debug(f"Enqueued conversation for user {user_id}, queue size: {self.queue.qsize()}")

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    def _worker(self):
        """Background worker that processes queue"""
        logger.info("Memory consolidation worker running")

        while self.running:
            try:
                # Get item from queue (blocks for 1 second)
                try:
                    item = self.queue.get(timeout=1.0)
                except Empty:
                    continue  # Queue empty, check if still running

                # Process the conversation
                user_id = item['user_id']
                conversation = item['conversation']

                logger.debug(f"Processing memory extraction for user {user_id}")

                try:
                    result = self.extraction_service.extract_from_conversation(
                        user_id=user_id,
                        conversation=conversation
                    )

                    episodic_count = len(result.get('episodic', []))
                    semantic_count = len(result.get('semantic', []))

                    if episodic_count > 0 or semantic_count > 0:
                        logger.info(
                            f"Extracted {episodic_count} episodic, "
                            f"{semantic_count} semantic memories for user {user_id}"
                        )
                except Exception as e:
                    logger.error(f"Memory extraction failed for user {user_id}: {e}", exc_info=True)

                self.queue.task_done()

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                continue

        logger.info("Memory consolidation worker exiting")

# Global instance (initialized in app startup)
_consolidation_queue = None

def get_consolidation_queue() -> MemoryConsolidationQueue:
    """Get global consolidation queue"""
    global _consolidation_queue
    if _consolidation_queue is None:
        raise RuntimeError("Consolidation queue not initialized. Call init_consolidation_queue() first.")
    return _consolidation_queue

def init_consolidation_queue(extraction_service: MemoryExtractionService) -> MemoryConsolidationQueue:
    """Initialize global queue (call from app startup)"""
    global _consolidation_queue
    if _consolidation_queue is not None:
        logger.warning("Consolidation queue already initialized")
        return _consolidation_queue

    _consolidation_queue = MemoryConsolidationQueue(extraction_service)
    _consolidation_queue.start()
    logger.info("Global consolidation queue initialized")
    return _consolidation_queue

def shutdown_consolidation_queue():
    """Shutdown global queue (call from app shutdown)"""
    global _consolidation_queue
    if _consolidation_queue is not None:
        _consolidation_queue.stop()
        _consolidation_queue = None
        logger.info("Global consolidation queue shutdown")
