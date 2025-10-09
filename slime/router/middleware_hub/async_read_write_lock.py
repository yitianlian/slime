"""High-performance async read-write lock implementation.
Provides better concurrency than threading.RLock for asyncio applications.
"""
import asyncio
from typing import List


class AsyncReadWriteLock:
    """
    High-performance asynchronous read-write lock.

    - Multiple readers can access concurrently
    - Only one writer can access at a time
    - Readers and writers are mutually exclusive
    - Simple implementation to avoid deadlocks
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the async read-write lock.

        Args:
            debug: Enable debug logging for lock operations
        """
        self._readers = 0
        self._writers = 0
        self._condition = asyncio.Condition()
        self._debug = debug

    async def acquire_read(self) -> None:
        """
        Acquire a read lock.
        Multiple readers can access simultaneously.
        Blocks only when there are active writers.
        """
        if self._debug:
            print(f"[RWLock] Reader attempting to acquire (readers: {self._readers}, writers: {self._writers})")

        async with self._condition:
            # Wait for all writers to finish
            while self._writers > 0:
                if self._debug:
                    print(f"[RWLock] Reader waiting for writer to finish")
                await self._condition.wait()

            self._readers += 1
            if self._debug:
                print(f"[RWLock] Reader acquired (readers: {self._readers}, writers: {self._writers})")

    async def release_read(self) -> None:
        """Release a read lock."""
        if self._debug:
            print(f"[RWLock] Reader releasing (readers: {self._readers}, writers: {self._writers})")

        async with self._condition:
            self._readers -= 1
            if self._readers == 0:
                # Notify waiting writers that all readers are done
                self._condition.notify_all()
            if self._debug:
                print(f"[RWLock] Reader released (readers: {self._readers}, writers: {self._writers})")

    async def acquire_write(self) -> None:
        """
        Acquire a write lock.
        Only one writer can access at a time.
        Blocks all other readers and writers.
        """
        if self._debug:
            print(f"[RWLock] Writer attempting to acquire (readers: {self._readers}, writers: {self._writers})")

        async with self._condition:
            # Wait for all readers and other writers to finish
            while self._writers > 0 or self._readers > 0:
                if self._debug:
                    print(f"[RWLock] Writer waiting (readers: {self._readers}, writers: {self._writers})")
                await self._condition.wait()

            self._writers += 1
            if self._debug:
                print(f"[RWLock] Writer acquired (readers: {self._readers}, writers: {self._writers})")

    async def release_write(self) -> None:
        """Release a write lock."""
        if self._debug:
            print(f"[RWLock] Writer releasing (readers: {self._readers}, writers: {self._writers})")

        async with self._condition:
            self._writers -= 1
            # Notify all waiting tasks (both readers and writers)
            self._condition.notify_all()
            if self._debug:
                print(f"[RWLock] Writer released (readers: {self._readers}, writers: {self._writers})")

    def get_stats(self) -> dict:
        """Get current lock statistics for debugging."""
        return {
            "readers": self._readers,
            "writers": self._writers,
        }


class ReadLockContext:
    """Context manager for read lock."""

    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_read()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_read()


class WriteLockContext:
    """Context manager for write lock."""

    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_write()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_write()


# Convenience context managers
def read_lock(lock: AsyncReadWriteLock) -> ReadLockContext:
    """Create a read lock context manager."""
    return ReadLockContext(lock)


def write_lock(lock: AsyncReadWriteLock) -> WriteLockContext:
    """Create a write lock context manager."""
    return WriteLockContext(lock)


# Simple lock for compatibility (doesn't distinguish read/write)
class AsyncLock:
    """Simple async lock for cases where read/write distinction isn't needed."""

    def __init__(self, debug: bool = False):
        self._lock = asyncio.Lock()
        self._debug = debug

    async def acquire(self) -> None:
        if self._debug:
            print("[AsyncLock] Acquiring")
        await self._lock.acquire()
        if self._debug:
            print("[AsyncLock] Acquired")

    async def release(self) -> None:
        if self._debug:
            print("[AsyncLock] Releasing")
        self._lock.release()
        if self._debug:
            print("[AsyncLock] Released")

    def __enter__(self):
        raise RuntimeError("Use 'async with' instead of 'with'")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()