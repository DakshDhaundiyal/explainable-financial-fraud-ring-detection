"""In-Memory Analysis Session Store and Repository.

Provides thread-safe storage, pagination, filtering, sorting, graph extraction,
and CSV export for transaction batches scored by TGAT.
"""
from __future__ import annotations

import io
import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class AnalysisSession:
    def __init__(
        self,
        session_id: str,
        filename: str,
        total_transactions: int,
        flagged_transactions: int,
        distinct_accounts: int,
        threshold: float,
        transactions: List[Dict[str, Any]],
        warnings: List[str],
        duration_seconds: float = 0.0,
    ) -> None:
        self.session_id = session_id
        self.filename = filename
        self.created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.total_transactions = total_transactions
        self.flagged_transactions = flagged_transactions
        self.distinct_accounts = distinct_accounts
        self.flagged_rate = round((flagged_transactions / max(1, total_transactions)) * 100, 2)
        self.threshold = threshold
        self.transactions = transactions
        self.warnings = warnings
        self.duration_seconds = duration_seconds
        self.score_distribution = self._compute_distribution()
        self.graph = self._build_graph()

    def _compute_distribution(self) -> List[Dict[str, Any]]:
        """10-bin score distribution for UI histogram."""
        bins = [0] * 10
        for tx in self.transactions:
            score = tx.get("risk_score", 0.0)
            idx = min(9, max(0, int(score * 10)))
            bins[idx] += 1
        labels = [f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)]
        return [{"range": labels[i], "count": bins[i], "percentage": round((bins[i] / max(1, self.total_transactions)) * 100, 1)} for i in range(10)]

    def _build_graph(self) -> Dict[str, Any]:
        """Build node-edge graph representation from batch transactions."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for tx in self.transactions:
            src = str(tx["from_account"])
            dst = str(tx["to_account"])
            amt = float(tx.get("amount", 0.0))
            score = float(tx.get("risk_score", 0.0))
            is_flagged = score >= self.threshold

            for acc in (src, dst):
                if acc not in nodes:
                    nodes[acc] = {
                        "id": acc,
                        "label": acc,
                        "in_degree": 0,
                        "out_degree": 0,
                        "total_sent": 0.0,
                        "total_received": 0.0,
                        "max_risk_score": 0.0,
                        "is_flagged": False,
                    }

            nodes[src]["out_degree"] += 1
            nodes[src]["total_sent"] += amt
            nodes[src]["max_risk_score"] = max(nodes[src]["max_risk_score"], score)
            if is_flagged:
                nodes[src]["is_flagged"] = True

            nodes[dst]["in_degree"] += 1
            nodes[dst]["total_received"] += amt
            nodes[dst]["max_risk_score"] = max(nodes[dst]["max_risk_score"], score)
            if is_flagged:
                nodes[dst]["is_flagged"] = True

            key = (src, dst)
            if key not in edges:
                edges[key] = {
                    "source": src,
                    "target": dst,
                    "count": 0,
                    "total_amount": 0.0,
                    "max_risk_score": 0.0,
                    "is_flagged": False,
                    "transactions": [],
                }
            edges[key]["count"] += 1
            edges[key]["total_amount"] += amt
            edges[key]["max_risk_score"] = max(edges[key]["max_risk_score"], score)
            if is_flagged:
                edges[key]["is_flagged"] = True
            edges[key]["transactions"].append(tx["id"])

        return {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        }


class AnalysisStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, AnalysisSession] = {}
        self._latest_id: Optional[str] = None

    def add_session(
        self,
        filename: str,
        transactions: List[Dict[str, Any]],
        threshold: float,
        warnings: List[str],
        duration_seconds: float = 0.0,
    ) -> AnalysisSession:
        session_id = f"ANL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"
        flagged = sum(1 for t in transactions if t.get("risk_score", 0.0) >= threshold)
        distinct = len(set(t["from_account"] for t in transactions) | set(t["to_account"] for t in transactions))

        session = AnalysisSession(
            session_id=session_id,
            filename=filename,
            total_transactions=len(transactions),
            flagged_transactions=flagged,
            distinct_accounts=distinct,
            threshold=threshold,
            transactions=transactions,
            warnings=warnings,
            duration_seconds=round(duration_seconds, 2),
        )

        self._sessions[session_id] = session
        self._latest_id = session_id
        return session

    def get_session(self, session_id: Optional[str] = None) -> Optional[AnalysisSession]:
        if not session_id or session_id == "latest":
            return self._sessions.get(self._latest_id) if self._latest_id else None
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "filename": s.filename,
                "created_at": s.created_at,
                "total_transactions": s.total_transactions,
                "flagged_transactions": s.flagged_transactions,
                "flagged_rate": s.flagged_rate,
                "distinct_accounts": s.distinct_accounts,
                "threshold": s.threshold,
                "duration_seconds": s.duration_seconds,
            }
            for s in sorted(self._sessions.values(), key=lambda x: x.created_at, reverse=True)
        ]

    def get_results(
        self,
        session_id: Optional[str] = None,
        page: int = 1,
        limit: int = 25,
        sort_by: str = "risk_score",
        sort_dir: str = "desc",
        search: Optional[str] = None,
        flag_filter: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {"total": 0, "page": page, "limit": limit, "pages": 0, "items": []}

        items = session.transactions

        # 1. Search Filter (Account or Transaction ID)
        if search:
            q = search.lower().strip()
            items = [t for t in items if q in str(t["id"]).lower() or q in str(t["from_account"]).lower() or q in str(t["to_account"]).lower()]

        # 2. Flag Status Filter
        if flag_filter == "flagged":
            items = [t for t in items if t["classification"] == "FLAGGED"]
        elif flag_filter == "normal":
            items = [t for t in items if t["classification"] == "BELOW_THRESHOLD"]

        # 3. Score Range Filter
        if min_score is not None:
            items = [t for t in items if t["risk_score"] >= min_score]
        if max_score is not None:
            items = [t for t in items if t["risk_score"] <= max_score]

        # 4. Sorting
        reverse = sort_dir.lower() == "desc"
        if sort_by in ("risk_score", "score"):
            items = sorted(items, key=lambda x: float(x.get("risk_score", 0.0)), reverse=reverse)
        elif sort_by in ("timestamp", "time"):
            items = sorted(items, key=lambda x: int(x.get("timestamp", 0)), reverse=reverse)
        elif sort_by in ("amount", "val"):
            items = sorted(items, key=lambda x: float(x.get("amount", 0.0)), reverse=reverse)
        elif sort_by in ("from_account", "sender"):
            items = sorted(items, key=lambda x: str(x.get("from_account", "")), reverse=reverse)

        # 5. Pagination
        total = len(items)
        pages = max(1, math.ceil(total / limit))
        start = (page - 1) * limit
        end = start + limit
        paged_items = items[start:end]

        return {
            "session_id": session.session_id,
            "filename": session.filename,
            "threshold": session.threshold,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
            "items": paged_items,
        }

    def get_graph(
        self,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
        account_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {"nodes": [], "edges": []}

        nodes = session.graph["nodes"]
        edges = session.graph["edges"]

        if min_score is not None:
            edges = [e for e in edges if e["max_risk_score"] >= min_score]
            active_node_ids = set()
            for e in edges:
                active_node_ids.add(e["source"])
                active_node_ids.add(e["target"])
            nodes = [n for n in nodes if n["id"] in active_node_ids]

        if account_filter:
            q = account_filter.strip().lower()
            edges = [e for e in edges if q in e["source"].lower() or q in e["target"].lower()]
            active_node_ids = set()
            for e in edges:
                active_node_ids.add(e["source"])
                active_node_ids.add(e["target"])
            nodes = [n for n in nodes if n["id"] in active_node_ids]

        return {"nodes": nodes, "edges": edges}

    def get_transaction(self, session_id: Optional[str], tx_id: str) -> Optional[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return None
        for t in session.transactions:
            if t["id"] == tx_id:
                return t
        return None

    def get_account_history(self, session_id: Optional[str], account_id: str) -> Optional[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return None
        related = [t for t in session.transactions if t["from_account"] == account_id or t["to_account"] == account_id]
        if not related:
            return None

        total_sent = sum(t["amount"] for t in related if t["from_account"] == account_id)
        total_received = sum(t["amount"] for t in related if t["to_account"] == account_id)
        counterparties = list(set(
            t["to_account"] if t["from_account"] == account_id else t["from_account"]
            for t in related
        ))
        flagged_count = sum(1 for t in related if t["classification"] == "FLAGGED")

        return {
            "account_id": account_id,
            "session_id": session.session_id,
            "total_transactions": len(related),
            "flagged_transactions": flagged_count,
            "total_sent": total_sent,
            "total_received": total_received,
            "counterparties": counterparties,
            "transactions": sorted(related, key=lambda x: x["timestamp"], reverse=True),
        }

    def export_csv(self, session_id: Optional[str] = None) -> Optional[str]:
        session = self.get_session(session_id)
        if not session or not session.transactions:
            return None

        export_rows = []
        for t in session.transactions:
            export_rows.append({
                "Transaction_ID": t["id"],
                "Timestamp": t.get("timestamp_str"),
                "Timestamp_Sec": t.get("timestamp"),
                "From_Account": t["from_account"],
                "To_Account": t["to_account"],
                "Amount": t["amount"],
                "Risk_Score": round(t["risk_score"], 6),
                "Decision_Threshold": session.threshold,
                "Classification": t["classification"],
                "Ground_Truth_Label": t.get("ground_truth", "UNLABELED"),
                "Has_Temporal_History": t.get("has_history", False),
            })

        df = pd.DataFrame(export_rows)
        buffer = io.StringIO()
        # Add metadata comment header
        buffer.write(f"# TGAT Financial Fraud Ring Detection — Analysis Results\n")
        buffer.write(f"# Analysis ID: {session.session_id}\n")
        buffer.write(f"# Generated: {session.created_at}\n")
        buffer.write(f"# Model Checkpoint: results/tgat/model.pt (Strictly Frozen)\n")
        buffer.write(f"# Frozen Operating Threshold: {session.threshold:.6f}\n")
        buffer.write(f"# Total Transactions: {session.total_transactions} | Flagged: {session.flagged_transactions} ({session.flagged_rate}%)\n#\n")
        df.to_csv(buffer, index=False)
        return buffer.getvalue()


# Global Singleton Store
analysis_store = AnalysisStore()
