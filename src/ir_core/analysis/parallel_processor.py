# src/ir_core/analysis/parallel_processor.py

"""
Parallel processing utilities for query analysis operations.

This module provides reusable parallel processing functionality
to eliminate code duplication across analysis components.
"""

from typing import List, Any, Callable, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ParallelProcessor:
    """
    Utility class for parallel processing of analysis operations.
    """

    def __init__(self, max_workers: Optional[int] = None, enable_parallel: bool = True):
        """
        Initialize the parallel processor.

        Args:
            max_workers: Maximum number of worker threads
            enable_parallel: Whether to enable parallel processing
        """
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel

    def process_batch(
        self,
        items: List[T],
        processor_func: Callable[[T], R],
        batch_threshold: int = 10,
        operation_name: str = "items"
    ) -> List[R]:
        """
        Process a batch of items with optional parallel processing.

        Args:
            items: List of items to process
            processor_func: Function to process each item
            batch_threshold: Minimum batch size for parallel processing
            operation_name: Name of the operation for logging

        Returns:
            List[R]: Results in original order
        """
        if not items:
            return []

        # Use parallel processing for larger batches
        if (len(items) > batch_threshold and
            self.enable_parallel and
            self.max_workers != 0):

            max_workers = self._get_max_workers(len(items))
            logger.info(f"🔄 Processing {len(items)} {operation_name} using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all processing tasks
                future_to_item = {executor.submit(processor_func, item): item for item in items}

                # Collect results in order
                results = []
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append((future_to_item[future], result))
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")
                        # Return None for failed items
                        results.append((future_to_item[future], None))

                # Sort results back to original order
                results.sort(key=lambda x: items.index(x[0]))
                return [result for _, result in results]
        else:
            # Use sequential processing for small batches
            return [processor_func(item) for item in items]

    def process_batch_with_error_handling(
        self,
        items: List[T],
        processor_func: Callable[[T], R],
        error_result: R,
        batch_threshold: int = 10,
        operation_name: str = "items"
    ) -> List[R]:
        """
        Process a batch of items with error handling and optional parallel processing.

        Args:
            items: List of items to process
            processor_func: Function to process each item
            error_result: Default result to return for failed items
            batch_threshold: Minimum batch size for parallel processing
            operation_name: Name of the operation for logging

        Returns:
            List[R]: Results in original order, with error_result for failures
        """
        if not items:
            return []

        # Use parallel processing for larger batches
        if (len(items) > batch_threshold and
            self.enable_parallel and
            self.max_workers != 0):

            max_workers = self._get_max_workers(len(items))
            logger.info(f"🔄 Processing {len(items)} {operation_name} using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all processing tasks
                future_to_item = {executor.submit(processor_func, item): item for item in items}

                # Collect results in order
                results = []
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append((future_to_item[future], result))
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")
                        # Return error result for failed items
                        results.append((future_to_item[future], error_result))

                # Sort results back to original order
                results.sort(key=lambda x: items.index(x[0]))
                return [result for _, result in results]
        else:
            # Use sequential processing for small batches
            results = []
            for item in items:
                try:
                    result = processor_func(item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    results.append(error_result)
            return results

    def _get_max_workers(self, item_count: int) -> int:
        """
        Determine the optimal number of workers for the given item count.

        Args:
            item_count: Number of items to process

        Returns:
            int: Optimal number of workers
        """
        if self.max_workers is not None:
            return min(self.max_workers, item_count)
        else:
            # Conservative default: min of 4 or item_count
            return min(4, item_count)