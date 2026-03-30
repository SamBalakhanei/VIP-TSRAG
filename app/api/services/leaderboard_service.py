from app.api.core.database import get_connection


class LeaderboardService:
    @staticmethod
    def get_leaderboard(
        metric_name: str = "MASE",
        family: str | None = None,
        sector: str | None = None,
        horizon: int | None = None,
        lookback: int | None = None,
    ):
        query = """
            SELECT
                m.model_id,
                m.model_name,
                m.family,
                t.task_name,
                t.horizon,
                t.lookback,
                t.sector,
                le.metric_name,
                le.mean_score,
                le.num_runs,
                le.num_series
            FROM leaderboard_entries le
            JOIN models m ON le.model_id = m.model_id
            JOIN tasks t ON le.task_id = t.task_id
            WHERE le.metric_name = ?
        """
        params = [metric_name]

        if family:
            query += " AND m.family = ?"
            params.append(family)

        if sector:
            query += " AND t.sector = ?"
            params.append(sector)

        if horizon is not None:
            query += " AND t.horizon = ?"
            params.append(horizon)

        if lookback is not None:
            query += " AND t.lookback = ?"
            params.append(lookback)

        query += " ORDER BY le.mean_score ASC"
        print("QUERY:", query)
        print("PARAMS:", params)

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        return rows