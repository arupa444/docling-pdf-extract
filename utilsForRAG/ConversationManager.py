from typing import Dict
from datetime import datetime, timedelta


class ConversationManager:
    def __init__(self):
        # {session_id: {"history": [], "last_access": datetime, "token_count": int}}
        self.sessions = {}
        self.MAX_TOKENS = 500
        self.SESSION_TTL = timedelta(hours=1)  # Auto-cleanup

    def _approximate_tokens(self, text: str) -> int:
        """Fast token approximation: ~0.75 tokens per word"""
        return int(len(text.split()) * 0.75)

    def _cleanup_expired(self):
        """Remove sessions older than TTL"""
        now = datetime.now()
        expired = [sid for sid, data in self.sessions.items()
                   if now - data["last_access"] > self.SESSION_TTL]
        for sid in expired:
            del self.sessions[sid]

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str):
        """Add user-assistant exchange and maintain token limit"""
        if len(self.sessions) > 1000:  # Prevent memory bloat
            self._cleanup_expired()

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "last_access": datetime.now(),
                "token_count": 0
            }

        session = self.sessions[session_id]
        session["last_access"] = datetime.now()

        # Calculate tokens for new exchange
        new_tokens = self._approximate_tokens(user_msg + assistant_msg)

        # Add new exchange
        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": assistant_msg})
        session["token_count"] += new_tokens

        # Truncate from front if over limit (FIFO)
        while session["token_count"] > self.MAX_TOKENS and len(session["history"]) > 2:
            removed = session["history"].pop(0)  # Remove oldest user message
            removed_assistant = session["history"].pop(0)  # Remove corresponding assistant
            session["token_count"] -= self._approximate_tokens(
                removed["content"] + removed_assistant["content"]
            )

    def get_context(self, session_id: str) -> str:
        """Get formatted context string (fast)"""
        if session_id not in self.sessions:
            return ""

        history = self.sessions[session_id]["history"]
        if not history:
            return ""

        # Format only recent exchanges (no need to format all if truncated)
        return "Previous conversation:\n" + "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        ) + "\n\n"

    def clear(self, session_id: str):
        """Clear session history"""
        self.sessions.pop(session_id, None)

    def get_stats(self, session_id: str) -> Dict:
        """Get session statistics"""
        if session_id not in self.sessions:
            return {"exists": False}

        session = self.sessions[session_id]
        return {
            "exists": True,
            "message_count": len(session["history"]),
            "token_count": session["token_count"],
            "last_access": session["last_access"].isoformat()
        }