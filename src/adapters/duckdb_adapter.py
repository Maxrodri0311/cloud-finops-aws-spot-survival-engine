"""
src/adapters/duckdb_adapter.py - Concrete DuckDB Analytical Storage Adapter.
Implements AnalyticalStorageProtocol.
Emulates AWS Athena federated queries and AWS Glue Data Catalog over local
Hive-partitioned Parquet files with automatic predicate pushdown and projection.
"""

import os
from typing import Optional, List, Dict, Any
import duckdb
import pandas as pd
from src.domain.contracts import AnalyticalStorageProtocol


class DuckDBAnalyticalAdapter(AnalyticalStorageProtocol):
    """
    Local-First In-Memory DuckDB OLAP Engine.
    Executes vectorized queries on Hive-partitioned Parquet data lakes.
    """

    def __init__(self, database: str = ":memory:"):
        self.conn = duckdb.connect(database)
        # Enable hive partitioning by default in DuckDB
        self.conn.execute("SET preserve_insertion_order=false;")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes an arbitrary SQL query returning a Pandas DataFrame."""
        return self.conn.execute(query).df()

    def scan_parquet(
        self,
        base_path: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Scans Hive-partitioned parquet files emulating AWS Athena partition projection.
        Example base_path: 'data/spot_events/**/*.parquet'
        """
        select_cols = ", ".join(columns) if columns else "*"
        where_clauses = []
        if filters:
            for col, val in filters.items():
                if isinstance(val, str):
                    where_clauses.append(f"{col} = '{val}'")
                elif isinstance(val, (int, float)):
                    where_clauses.append(f"{col} = {val}")
                elif isinstance(val, list):
                    formatted_vals = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in val])
                    where_clauses.append(f"{col} IN ({formatted_vals})")

        where_stmt = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT {select_cols}
            FROM read_parquet('{base_path}', hive_partitioning = true)
            {where_stmt};
        """
        return self.execute_query(query)

    def register_view(self, view_name: str, parquet_glob: str) -> None:
        """Registers a persistent virtual view over partitioned Parquet files."""
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning = true);
        """)

    def table_exists(self, table_name: str) -> bool:
        """Checks if a table or view exists in the DuckDB catalog."""
        res = self.conn.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}';"
        ).fetchone()
        return bool(res and res[0] > 0)
