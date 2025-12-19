"""Threading utilities for running backend operations in GUI.

This module provides utilities for executing async operations in background threads
and safely updating the GUI from those threads.
"""

import asyncio
import threading
from typing import Callable, Any, Optional
from functools import wraps


def run_async_in_thread(
    async_func: Callable,
    on_success: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    *args,
    **kwargs
) -> threading.Thread:
    """Execute an async function in a background thread.
    
    This function creates a new event loop in a background thread and runs
    the async function in that loop. Callbacks are executed in the same thread.
    
    Args:
        async_func: Async function to execute
        on_success: Callback to execute on success with result
        on_error: Callback to execute on error with exception
        *args: Positional arguments for async_func
        **kwargs: Keyword arguments for async_func
        
    Returns:
        The created thread (already started)
        
    Example:
        >>> async def fetch_data():
        ...     return await some_async_operation()
        >>> 
        >>> def handle_result(result):
        ...     print(f"Got result: {result}")
        >>> 
        >>> thread = run_async_in_thread(fetch_data, on_success=handle_result)
    """
    def thread_target():
        """Target function for the thread."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async function
                result = loop.run_until_complete(async_func(*args, **kwargs))
                
                # Call success callback if provided
                if on_success:
                    on_success(result)
                    
            finally:
                loop.close()
                
        except Exception as e:
            # Call error callback if provided
            if on_error:
                on_error(e)
            else:
                # Re-raise if no error handler
                raise
    
    # Create and start thread
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def safe_gui_update(root, callback: Callable, *args, **kwargs) -> None:
    """Schedule a GUI update to run in the main thread.
    
    This function uses tkinter's after() method to schedule a callback
    to run in the main GUI thread. This is necessary when updating GUI
    elements from background threads.
    
    Args:
        root: Tkinter root window or any widget
        callback: Function to execute in main thread
        *args: Positional arguments for callback
        **kwargs: Keyword arguments for callback
        
    Example:
        >>> def update_label(text):
        ...     label.config(text=text)
        >>> 
        >>> # From background thread:
        >>> safe_gui_update(root, update_label, "New text")
    """
    def wrapper():
        callback(*args, **kwargs)
    
    root.after(0, wrapper)


def async_task(on_success: Optional[Callable] = None, on_error: Optional[Callable] = None):
    """Decorator to run async functions in background threads from GUI.
    
    This decorator wraps an async function to automatically run it in a
    background thread when called from GUI code.
    
    Args:
        on_success: Optional callback for successful completion
        on_error: Optional callback for errors
        
    Returns:
        Decorator function
        
    Example:
        >>> @async_task(on_success=lambda r: print(f"Done: {r}"))
        >>> async def long_operation():
        ...     await asyncio.sleep(5)
        ...     return "Complete"
        >>> 
        >>> # Call from GUI event handler:
        >>> long_operation()  # Runs in background
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return run_async_in_thread(func, on_success, on_error, *args, **kwargs)
        return wrapper
    return decorator
