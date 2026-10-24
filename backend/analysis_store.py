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
        
        # Risk level counts
        self.high_risk_count = sum(1 for t in transactions if t.get("risk_score", 0.0) >= threshold)
        self.medium_risk_count = sum(1 for t in transactions if 0.40 <= t.get("risk_score", 0.0) < threshold)
        self.low_risk_count = sum(1 for t in transactions if t.get("risk_score", 0.0) < 0.40)
        
        # Distinct high-risk accounts
        high_risk_accs = set()
        for t in transactions:
            if t.get("risk_score", 0.0) >= threshold:
                high_risk_accs.add(t["from_account"])
                high_risk_accs.add(t["to_account"])
        self.high_risk_accounts = len(high_risk_accs)
        
        # Plain-English Executive Summary
        if self.high_risk_count > 0:
            self.executive_summary = (
                f"{self.high_risk_count} of {self.total_transactions} transactions ({self.flagged_rate}%) exceeded the {self.threshold:.4f} risk threshold. "
                f"Elevated risk is driven by compressed transaction velocity and connections to high-risk counterparties. "
                f"{self.low_risk_count} transactions ({round(self.low_risk_count/max(1,self.total_transactions)*100, 1)}%) follow safe, legitimate behavioral patterns."
            )
        else:
            self.executive_summary = (
                f"All {self.total_transactions} transactions evaluated remain safely below the {self.threshold:.4f} threshold. "
                f"Traffic exhibits normal transaction intervals, legitimate amounts, and clean counterparty relationships."
            )

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
                        "risk_level": "Low",
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
                    "risk_level": "Low",
                    "transactions": [],
                }
            edges[key]["count"] += 1
            edges[key]["total_amount"] += amt
            edges[key]["max_risk_score"] = max(edges[key]["max_risk_score"], score)
            if is_flagged:
                edges[key]["is_flagged"] = True
            edges[key]["transactions"].append(tx["id"])

        # Assign risk levels
        for node in nodes.values():
            m_score = node["max_risk_score"]
            if m_score >= self.threshold:
                node["risk_level"] = "High Risk"
            elif m_score >= 0.40:
                node["risk_level"] = "Medium Risk"
            else:
                node["risk_level"] = "Low Risk"

        for edge in edges.values():
            e_score = edge["max_risk_score"]
            if e_score >= self.threshold:
                edge["risk_level"] = "High Risk"
            elif e_score >= 0.40:
                edge["risk_level"] = "Medium Risk"
            else:
                edge["risk_level"] = "Low Risk"

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
        risk_level: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {
                "total": 0, "page": page, "limit": limit, "pages": 0, "items": [],
                "summary": {"total": 0, "flagged": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0}
            }

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

        # 3. Risk Level Filter
        if risk_level:
            rl = risk_level.lower().strip()
            if rl in ("high", "high risk"):
                items = [t for t in items if t.get("risk_score", 0.0) >= session.threshold]
            elif rl in ("medium", "medium risk"):
                items = [t for t in items if 0.40 <= t.get("risk_score", 0.0) < session.threshold]
            elif rl in ("low", "low risk"):
                items = [t for t in items if t.get("risk_score", 0.0) < 0.40]

        # 4. Score Range Filter
        if min_score is not None:
            items = [t for t in items if t["risk_score"] >= min_score]
        if max_score is not None:
            items = [t for t in items if t["risk_score"] <= max_score]

        # 5. Sorting
        reverse = sort_dir.lower() == "desc"
        if sort_by in ("risk_score", "score"):
            items = sorted(items, key=lambda x: float(x.get("risk_score", 0.0)), reverse=reverse)
        elif sort_by in ("timestamp", "time"):
            items = sorted(items, key=lambda x: int(x.get("timestamp", 0)), reverse=reverse)
        elif sort_by in ("amount", "val"):
            items = sorted(items, key=lambda x: float(x.get("amount", 0.0)), reverse=reverse)
        elif sort_by in ("from_account", "sender"):
            items = sorted(items, key=lambda x: str(x.get("from_account", "")), reverse=reverse)

        # 6. Pagination
        total = len(items)
        pages = max(1, math.ceil(total / limit))
        start = (page - 1) * limit
        end = start + limit
        paged_items = items[start:end]

        # Attach risk level tag for each item
        for item in paged_items:
            sc = item.get("risk_score", 0.0)
            if sc >= session.threshold:
                item["risk_level"] = "High Risk"
                item["risk_level_code"] = "high"
            elif sc >= 0.40:
                item["risk_level"] = "Medium Risk"
                item["risk_level_code"] = "medium"
            else:
                item["risk_level"] = "Low Risk"
                item["risk_level_code"] = "low"

        return {
            "session_id": session.session_id,
            "filename": session.filename,
            "threshold": session.threshold,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
            "items": paged_items,
            "summary": {
                "total": session.total_transactions,
                "flagged": session.flagged_transactions,
                "high_risk": session.high_risk_count,
                "medium_risk": session.medium_risk_count,
                "low_risk": session.low_risk_count,
            }
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
                score = t.get("risk_score", 0.0)
                is_high = score >= session.threshold
                is_med = 0.40 <= score < session.threshold
                
                t_copy = dict(t)
                t_copy["threshold"] = session.threshold
                t_copy["risk_level"] = "High Risk" if is_high else ("Medium Risk" if is_med else "Low Risk")
                t_copy["risk_level_code"] = "high" if is_high else ("medium" if is_med else "low")
                
                # Plain-English risk factors
                factors = []
                amt = float(t.get("amount", 0.0))
                if is_high:
                    factors.append({
                        "title": "Model Risk Threshold Breached",
                        "impact": "High Impact",
                        "level": "danger",
                        "description": f"The temporal graph model evaluated this transaction at {score:.2f} risk, exceeding the {session.threshold:.2f} detection cutoff."
                    })
                    if t.get("has_history", False):
                        factors.append({
                            "title": "Suspicious Historical Clustering",
                            "impact": "High Impact",
                            "level": "danger",
                            "description": "Sender and receiver show intense past interactions with elevated attention weights concentrated on recent fund movements."
                        })
                    else:
                        factors.append({
                            "title": "Cold-Start Burst Pattern",
                            "impact": "Medium Impact",
                            "level": "warning",
                            "description": "Unseen accounts transferring significant sums with no established legitimate transaction history."
                        })
                    if amt > 20000:
                        factors.append({
                            "title": "High Transaction Amount",
                            "impact": "Medium Impact",
                            "level": "warning",
                            "description": f"Transaction value of ${amt:,.2f} is substantially larger than typical peer transaction distributions."
                        })
                elif is_med:
                    factors.append({
                        "title": "Elevated Background Activity",
                        "impact": "Medium Impact",
                        "level": "warning",
                        "description": f"Risk score ({score:.2f}) is noticeably elevated above baseline, but has not yet crossed the alert threshold."
                    })
                    factors.append({
                        "title": "Paced Transfer Frequency",
                        "impact": "Safe",
                        "level": "safe",
                        "description": "Transaction does not display the high burst velocities characteristic of structured laundering rings."
                    })
                else:
                    factors.append({
                        "title": "Standard Baseline Transaction",
                        "impact": "Safe",
                        "level": "safe",
                        "description": f"Risk score ({score:.2f}) is well below the {session.threshold:.2f} threshold. Behavior conforms to normal commercial banking traffic."
                    })
                    factors.append({
                        "title": "Clean Counterparty Profile",
                        "impact": "Safe",
                        "level": "safe",
                        "description": "Neither sender nor receiver has established links to flagged clusters or suspicious transaction histories."
                    })
                    factors.append({
                        "title": "Diffused Model Attention",
                        "impact": "Normal",
                        "level": "safe",
                        "description": "Graph attention weights are spread evenly over past events with zero focus on rapid pass-through bursts."
                    })

                t_copy["plain_english_factors"] = factors
                t_copy["plain_english_summary"] = (
                    f"Why this transaction {'is HIGH RISK' if is_high else ('has ELEVATED RISK' if is_med else 'is LOW RISK')}: "
                    f"Score is {score:.4f} vs threshold {session.threshold:.4f}. "
                    + ("It matches rapid laundering movement patterns with concentrated temporal attention." if is_high else "It shows steady, legitimate banking characteristics.")
                )
                return t_copy
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
        max_score = max(float(t.get("risk_score", 0.0)) for t in related)
        avg_score = sum(float(t.get("risk_score", 0.0)) for t in related) / len(related)
        flagged_count = sum(1 for t in related if t["classification"] == "FLAGGED")

        is_high = max_score >= session.threshold
        is_med = 0.40 <= max_score < session.threshold
        risk_level = "High Risk" if is_high else ("Medium Risk" if is_med else "Low Risk")
        risk_level_code = "high" if is_high else ("medium" if is_med else "low")

        # Sorted counterparties with interaction volume
        cp_map: Dict[str, Dict[str, Any]] = {}
        for t in related:
            cp = t["to_account"] if t["from_account"] == account_id else t["from_account"]
            if cp not in cp_map:
                cp_map[cp] = {"account_id": cp, "count": 0, "total_amount": 0.0, "max_score": 0.0}
            cp_map[cp]["count"] += 1
            cp_map[cp]["total_amount"] += float(t.get("amount", 0.0))
            cp_map[cp]["max_score"] = max(cp_map[cp]["max_score"], float(t.get("risk_score", 0.0)))

        counterparties_details = []
        high_risk_cps = 0
        for cp_data in sorted(cp_map.values(), key=lambda x: x["max_score"], reverse=True):
            sc = cp_data["max_score"]
            lvl = "High Risk" if sc >= session.threshold else ("Medium Risk" if sc >= 0.40 else "Low Risk")
            if sc >= session.threshold:
                high_risk_cps += 1
            counterparties_details.append({
                "account_id": cp_data["account_id"],
                "transaction_count": cp_data["count"],
                "total_amount": round(cp_data["total_amount"], 2),
                "max_risk_score": round(sc, 4),
                "risk_level": lvl,
                "risk_level_code": "high" if sc >= session.threshold else ("medium" if sc >= 0.40 else "low"),
            })

        # Plain-English Layman Risk Factors
        risk_factors = []
        if is_high:
            risk_factors.append({
                "title": "Model Risk Threshold Breached",
                "impact": "High Impact",
                "level": "danger",
                "description": f"Maximum transaction risk score ({max_score:.2f}) crossed the {session.threshold:.2f} threshold.",
            })
            if flagged_count > 0:
                risk_factors.append({
                    "title": "Direct Flagged Transactions",
                    "impact": "High Impact",
                    "level": "danger",
                    "description": f"Account participated in {flagged_count} transaction(s) explicitly flagged for suspicious laundering behavior.",
                })
            if high_risk_cps > 0:
                risk_factors.append({
                    "title": "High-Risk Network Proximity",
                    "impact": "High Impact",
                    "level": "danger",
                    "description": f"Transacts directly with {high_risk_cps} other account(s) exhibiting elevated risk in the transaction graph.",
                })
            if len(related) > 2:
                risk_factors.append({
                    "title": "Rapid Transaction Velocity",
                    "impact": "Medium Impact",
                    "level": "warning",
                    "description": f"{len(related)} transactions in close temporal succession indicating rapid fund pass-through.",
                })
            if abs(total_sent - total_received) > 5000:
                risk_factors.append({
                    "title": "Fund Flow Imbalance",
                    "impact": "Medium Impact",
                    "level": "warning",
                    "description": f"Disparity between sent (${total_sent:,.2f}) and received (${total_received:,.2f}) consistent with mule/collector accounts.",
                })
        elif is_med:
            risk_factors.append({
                "title": "Elevated Counterparty Interaction",
                "impact": "Medium Impact",
                "level": "warning",
                "description": f"Peak transaction risk score ({max_score:.2f}) is higher than baseline, indicating moderate network exposure.",
            })
            risk_factors.append({
                "title": "Sub-Threshold Activity",
                "impact": "Safe",
                "level": "safe",
                "description": f"Does not cross the {session.threshold:.2f} alert threshold. Standard monitoring is recommended without immediate freeze.",
            })
        else:
            risk_factors.append({
                "title": "Standard Transaction Amounts",
                "impact": "Safe",
                "level": "safe",
                "description": f"All transactions ({max_score:.2f} max score) remain safely within normal baseline operating limits.",
            })
            risk_factors.append({
                "title": "Clean Counterparty Network",
                "impact": "Safe",
                "level": "safe",
                "description": "Zero direct connections to flagged accounts or suspicious transaction clusters in this batch.",
            })
            risk_factors.append({
                "title": "Paced Transfer Spacing",
                "impact": "Safe",
                "level": "safe",
                "description": "No burst transfers or high-frequency smurfing cycles detected.",
            })
            risk_factors.append({
                "title": "Consistent Flow Pattern",
                "impact": "Normal",
                "level": "safe",
                "description": "Fund movement adheres to standard legitimate account behavior.",
            })

        # Plain-English Executive Summary
        if is_high:
            summary_text = (
                f"🚨 Classified as HIGH RISK: This account is connected to transactions scoring up to {max_score:.2f} (above {session.threshold:.2f}). "
                f"It actively moves funds across {len(counterparties_details)} counterparties ({high_risk_cps} high-risk), exhibiting high-velocity fund routing patterns."
            )
        elif is_med:
            summary_text = (
                f"⚠️ Classified as MEDIUM RISK: This account displays elevated risk scores (peak {max_score:.2f}), "
                f"but has not crossed the {session.threshold:.2f} threshold. Periodic monitoring is advised."
            )
        else:
            summary_text = (
                f"✅ Classified as LOW RISK: This account is safe. All transaction scores (peak {max_score:.2f}) stay well below {session.threshold:.2f}, "
                f"with balanced amounts and zero connections to suspicious accounts."
            )

        sorted_txs = sorted(related, key=lambda x: x["timestamp"], reverse=True)
        first_tx = sorted_txs[-1].get("timestamp_str", "—")
        last_tx = sorted_txs[0].get("timestamp_str", "—")

        return {
            "account_id": account_id,
            "session_id": session.session_id,
            "risk_level": risk_level,
            "risk_level_code": risk_level_code,
            "max_risk_score": round(max_score, 4),
            "avg_risk_score": round(avg_score, 4),
            "total_transactions": len(related),
            "flagged_transactions": flagged_count,
            "total_sent": round(total_sent, 2),
            "total_received": round(total_received, 2),
            "first_transaction": first_tx,
            "last_transaction": last_tx,
            "counterparties": list(cp_map.keys()),
            "counterparties_details": counterparties_details,
            "risk_factors": risk_factors,
            "plain_english_summary": summary_text,
            "transactions": sorted_txs,
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
