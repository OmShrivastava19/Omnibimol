from typing import Optional, Dict
import sqlite3
import json
from datetime import datetime

# cache_manager.py - SQLite caching with 24h expiry
class CacheManager:
    """Manages SQLite cache for API responses with 24-hour expiration"""
    
    def __init__(self, db_path: str = "omnibiomol_cache.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize cache database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached data if not expired (24h TTL)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value, timestamp FROM cache WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            value, timestamp = result
            # Check if cache is still valid (24 hours)
            if datetime.now().timestamp() - timestamp < 86400:  # 24h in seconds
                return json.loads(value)
            else:
                # Expired, delete it
                self.delete(key)
        return None
    
    def set(self, key: str, value: Dict):
        """Store data in cache with current timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache (key, value, timestamp)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now().timestamp()))
        
        conn.commit()
        conn.close()
    
    def delete(self, key: str):
        """Remove specific cache entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    
    def clear_expired(self):
        """Remove all expired entries (maintenance)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = datetime.now().timestamp() - 86400
        cursor.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff,))
        
        conn.commit()
        conn.close()